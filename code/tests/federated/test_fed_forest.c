/**
 * @file test_fed_forest.c
 * @brief Level 1, 2 & 3 tests for ht_fed_proto and Mode 1 Fed-Forest (ht_fed_forest).
 */

#include "test_harness.h"
#include "ht_fed_proto.h"
#include "ht_fed_forest.h"
#include <string.h>

static void test_fed_proto_framing_and_crc32(void) {
    const uint8_t sample_payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t buf[128];
    size_t written = 0;

    uint16_t flags = ht_fed_make_flags(HT_TASK_CLASSIFICATION);
    HT_ASSERT_EQ(ht_fed_write_frame(buf, sizeof(buf), HT_FED_MSG_FOREST_CLIENT_TREE,
                                    flags, 7U, 3U, 11U, sample_payload, 8U, &written), HT_OK);
    HT_ASSERT_EQ(written, sizeof(ht_fed_header_t) + 8U);

    ht_fed_header_t hdr;
    const uint8_t *out_payload = NULL;
    HT_ASSERT_EQ(ht_fed_parse_frame(buf, written, &hdr, &out_payload), HT_OK);
    HT_ASSERT_EQ(hdr.client_id, 7U);
    HT_ASSERT_EQ(hdr.round_id, 3U);
    HT_ASSERT_EQ(hdr.topology_version, 11U);
    HT_ASSERT_EQ(hdr.payload_len, 8U);
    HT_ASSERT_EQ(memcmp(out_payload, sample_payload, 8U), 0);

    /* Bit flip corruption must be rejected by CRC32 */
    buf[sizeof(ht_fed_header_t) + 2] ^= 0x55U;
    HT_ASSERT_EQ(ht_fed_parse_frame(buf, written, &hdr, &out_payload), HT_ERR_CORRUPT_IMAGE);
}

static void test_fed_forest_server_policies_and_exclude_self(void) {
    static ht_fed_forest_client_t client;
    static ht_fed_forest_server_t server;
    static uint8_t wire[32768];

    ht_fed_forest_server_cfg_t scfg;
    ht_fed_forest_default_server_config(&scfg, HT_TASK_CLASSIFICATION);
    scfg.agg_policy = HT_FED_FOREST_AGG_TOP_M;
    scfg.max_forest_size = 2U;
    HT_ASSERT_EQ(ht_fed_forest_server_init(&server, &scfg), HT_OK);

    ht_config_params_t tp;
    ht_default_params(&tp, 2, 2);

    /* Ingest 4 clients with distinct reported errors: id 1 (0.40), id 2 (0.05), id 3 (0.20), id 4 (0.10) */
    const ht_float_t errs[4] = {HT_CONST(0.40), HT_CONST(0.05), HT_CONST(0.20), HT_CONST(0.10)};
    for (uint16_t cid = 1U; cid <= 4U; ++cid) {
        ht_fed_forest_client_cfg_t ccfg;
        ht_fed_forest_default_client_config(&ccfg, cid, &tp);
        HT_ASSERT_EQ(ht_fed_forest_client_init(&client, &ccfg), HT_OK);
        client.member_faded_error[0] = errs[cid - 1U];

        size_t w = 0;
        HT_ASSERT_EQ(ht_fed_forest_client_export_update(&client, wire, sizeof(wire), &w), HT_OK);
        HT_ASSERT_EQ(ht_fed_forest_server_ingest(&server, wire, w), HT_OK);
    }
    HT_ASSERT_EQ(server.n_entries, 4U);

    /* Build TOP_M=2 excluding client 2 (best): should select client 4 (0.10) and client 3 (0.20) */
    size_t bw = 0;
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&server, 2U, true, wire, sizeof(wire), &bw), HT_OK);

    ht_fed_forest_client_cfg_t ccfg2;
    ht_fed_forest_default_client_config(&ccfg2, 2U, &tp);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&client, &ccfg2), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&client, wire, bw), HT_OK);
    HT_ASSERT_EQ(client.n_peers, 2U);
    HT_ASSERT_EQ(client.peer_client_ids[0], 4U);
    HT_ASSERT_EQ(client.peer_client_ids[1], 3U);

    /* Switch to ROUND_ROBIN and advance round to verify rotation */
    server.cfg.agg_policy = HT_FED_FOREST_AGG_ROUND_ROBIN;
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&server, 99U, false, wire, sizeof(wire), &bw), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&client, wire, bw), HT_OK);
    uint16_t first_id_r0 = client.peer_client_ids[0];

    ht_fed_forest_server_advance_round(&server);
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&server, 99U, false, wire, sizeof(wire), &bw), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&client, wire, bw), HT_OK);
    HT_ASSERT_NEQ(client.peer_client_ids[0], first_id_r0);
}

static void test_fed_forest_non_iid_collaboration_gain(void) {
    /* Client 0 only sees class 0 & 1 separated around x0=2 vs x0=5.
     * Client 1 only sees class 2 & 3 separated around x0=8 vs x0=11.
     * After Fed-Forest exchange, Client 0 (evaluating on all 4 classes) must dramatically improve. */
    static ht_fed_forest_client_t c0, c1;
    static ht_fed_forest_server_t srv;
    static uint8_t wire[32768];

    ht_config_params_t tp;
    ht_default_params(&tp, 2, 4);
    tp.grace_period = 20U;
    tp.delta = HT_CONST(1e-2);
    tp.leaf_prediction = HT_LEAF_MC;

    ht_fed_forest_client_cfg_t cfg0, cfg1;
    ht_fed_forest_default_client_config(&cfg0, 0U, &tp);
    ht_fed_forest_default_client_config(&cfg1, 1U, &tp);
    cfg0.voting = HT_VOTE_UNIFORM;
    cfg1.voting = HT_VOTE_UNIFORM;
    HT_ASSERT_EQ(ht_fed_forest_client_init(&c0, &cfg0), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&c1, &cfg1), HT_OK);

    ht_fed_forest_server_cfg_t scfg;
    ht_fed_forest_default_server_config(&scfg, HT_TASK_CLASSIFICATION);
    scfg.agg_policy = HT_FED_FOREST_AGG_ALL;
    HT_ASSERT_EQ(ht_fed_forest_server_init(&srv, &scfg), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 300; ++i) {
        ht_float_t jitter = HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.1));
        /* Client 0: classes 0 (x0 ~ 1) and 1 (x0 ~ 4) */
        uint8_t y0 = (uint8_t)(i % 2);
        x[0].num = (y0 == 0U) ? HT_ADD(HT_CONST(1.0), jitter) : HT_ADD(HT_CONST(4.0), jitter);
        x[1].num = HT_FROM_INT(i % 7);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one(&c0, x, y0, HT_CONST(1.0)), HT_OK);

        /* Client 1: classes 2 (x0 ~ 8) and 3 (x0 ~ 12) */
        uint8_t y1 = (uint8_t)(2 + (i % 2));
        x[0].num = (y1 == 2U) ? HT_ADD(HT_CONST(8.0), jitter) : HT_ADD(HT_CONST(12.0), jitter);
        x[1].num = HT_FROM_INT(i % 7);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one(&c1, x, y1, HT_CONST(1.0)), HT_OK);
    }

    /* Evaluate Client 0 BEFORE federation on classes 2 and 3 (unseen locally -> 0% on c2/c3) */
    int correct_before = 0;
    const ht_float_t centers[4] = {HT_CONST(1.2), HT_CONST(4.2), HT_CONST(8.2), HT_CONST(12.2)};
    for (uint8_t c = 0; c < 4U; ++c) {
        x[0].num = centers[c];
        x[1].num = HT_CONST(2.0);
        if (ht_fed_forest_client_predict(&c0, x) == c) {
            correct_before++;
        }
    }
    HT_ASSERT(correct_before <= 2);

    /* Exchange trees via server */
    size_t w0 = 0, w1 = 0, wb = 0;
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(&c0, wire, sizeof(wire), &w0), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w0), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(&c1, wire, sizeof(wire), &w1), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w1), HT_OK);

    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 0U, true, wire, sizeof(wire), &wb), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&c0, wire, wb), HT_OK);
    HT_ASSERT_EQ(c0.n_peers, 1U);

    /* Evaluate probabilities on classes 2 and 3: now non-zero and informative */
    ht_float_t probas[HT_MAX_CLASSES];
    x[0].num = HT_CONST(8.2);
    HT_ASSERT_EQ(ht_fed_forest_client_predict_proba(&c0, x, probas), HT_OK);
    HT_ASSERT(probas[2] > HT_CONST(0.3));

    x[0].num = HT_CONST(12.2);
    HT_ASSERT_EQ(ht_fed_forest_client_predict_proba(&c0, x, probas), HT_OK);
    HT_ASSERT(probas[3] > HT_CONST(0.3));
}

static void test_fed_forest_regression_and_drift_downweighting(void) {
#ifdef HT_CLASSIF_ONLY
    static ht_fed_forest_client_t c_reg;
    ht_config_params_t tp;
    ht_default_reg_params(&tp, 2);
    ht_fed_forest_client_cfg_t cfg;
    ht_fed_forest_default_client_config(&cfg, 1U, &tp);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&c_reg, &cfg), HT_ERR_INVALID_ARG);
#else
    static ht_fed_forest_client_t c_good, c_stale;
    static ht_fed_forest_server_t srv;
    static uint8_t wire[32768];

    ht_config_params_t tp;
    ht_default_reg_params(&tp, 2);
    tp.grace_period = 20U;
    tp.delta = HT_CONST(1e-2);
    tp.leaf_prediction = HT_LEAF_MEAN;

    ht_fed_forest_client_cfg_t cfg_g, cfg_s;
    ht_fed_forest_default_client_config(&cfg_g, 1U, &tp);
    ht_fed_forest_default_client_config(&cfg_s, 2U, &tp);
    cfg_g.voting = HT_VOTE_WEIGHTED;
    cfg_g.error_decay = HT_CONST(0.85);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&c_good, &cfg_g), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&c_stale, &cfg_s), HT_OK);

    ht_fed_forest_server_cfg_t scfg;
    ht_fed_forest_default_server_config(&scfg, HT_TASK_REGRESSION);
    HT_ASSERT_EQ(ht_fed_forest_server_init(&srv, &scfg), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Train c_stale on inverted/shifted target (y = 100.0) and c_good on true target (y = 10.0) */
    for (int i = 0; i < 120; ++i) {
        x[0].num = HT_FROM_INT(i % 10);
        x[1].num = HT_FROM_INT((i * 3) % 7);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one_reg(&c_good, x, HT_CONST(10.0), HT_CONST(1.0)), HT_OK);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one_reg(&c_stale, x, HT_CONST(100.0), HT_CONST(1.0)), HT_OK);
    }

    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(&c_stale, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 1U, true, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&c_good, wire, w), HT_OK);

    /* Feed 40 post-exchange samples at y = 10.0: weighted voting must down-weight stale peer */
    for (int i = 0; i < 40; ++i) {
        x[0].num = HT_FROM_INT(i % 10);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one_reg(&c_good, x, HT_CONST(10.0), HT_CONST(1.0)), HT_OK);
    }
    ht_float_t pred = ht_fed_forest_client_predict_reg(&c_good, x);
    HT_ASSERT_FLOAT_NEAR(pred, HT_CONST(10.0), HT_CONST(1.0));
#endif
}

static void test_fed_proto_corrupt_frames_and_fixed_flag(void) {
    HT_ASSERT_EQ(ht_fed_crc32(NULL, 10U), 0U);

    uint8_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t wire[128];
    size_t req = 0, w = 0;
    uint16_t flags = ht_fed_make_flags(HT_TASK_CLASSIFICATION);

    /* Query size & buffer-too-small paths */
    HT_ASSERT_EQ(ht_fed_write_frame(NULL, 0U, HT_FED_MSG_FOREST_CLIENT_TREE, flags, 1U, 1U, 0U, payload, 8U, &req), HT_OK);
    HT_ASSERT_EQ(req, sizeof(ht_fed_header_t) + 8U);
    HT_ASSERT_EQ(ht_fed_write_frame(wire, 8U, HT_FED_MSG_FOREST_CLIENT_TREE, flags, 1U, 1U, 0U, payload, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);
    HT_ASSERT_EQ(ht_fed_finalize_frame(NULL, 0U, HT_FED_MSG_FOREST_CLIENT_TREE, flags, 1U, 1U, 0U, 8U, &req), HT_OK);
    HT_ASSERT_EQ(ht_fed_finalize_frame(wire, 8U, HT_FED_MSG_FOREST_CLIENT_TREE, flags, 1U, 1U, 0U, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_FOREST_CLIENT_TREE, flags, 1U, 1U, 0U, payload, 8U, &w), HT_OK);

    ht_fed_header_t hdr;
    const uint8_t *pl = NULL;
    HT_ASSERT_EQ(ht_fed_parse_frame(NULL, w, &hdr, &pl), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_parse_frame(wire, 4U, &hdr, &pl), HT_ERR_CORRUPT_IMAGE);

    /* Corrupt magic */
    uint8_t bad[128];
    memcpy(bad, wire, w);
    bad[0] ^= 0xFFU;
    HT_ASSERT_EQ(ht_fed_parse_frame(bad, w, &hdr, &pl), HT_ERR_CORRUPT_IMAGE);

    /* Corrupt protocol version */
    memcpy(bad, wire, w);
    ht_fed_header_t ch;
    memcpy(&ch, bad, sizeof(ch));
    ch.proto_version = 99U;
    memcpy(bad, &ch, sizeof(ch));
    HT_ASSERT_EQ(ht_fed_parse_frame(bad, w, &hdr, &pl), HT_ERR_CORRUPT_IMAGE);

    /* Cross-precision flag mismatch (flip HT_FED_FLAG_FIXED) */
    memcpy(bad, wire, w);
    memcpy(&ch, bad, sizeof(ch));
    ch.flags ^= HT_FED_FLAG_FIXED;
    memcpy(bad, &ch, sizeof(ch));
    HT_ASSERT_EQ(ht_fed_parse_frame(bad, w, &hdr, &pl), HT_ERR_CORRUPT_IMAGE);

    /* Truncated payload relative to payload_len */
    HT_ASSERT_EQ(ht_fed_parse_frame(wire, w - 2U, &hdr, &pl), HT_ERR_CORRUPT_IMAGE);

    /* Bit-flip CRC failure */
    memcpy(bad, wire, w);
    bad[sizeof(ht_fed_header_t)] ^= 0x01U;
    HT_ASSERT_EQ(ht_fed_parse_frame(bad, w, &hdr, &pl), HT_ERR_CORRUPT_IMAGE);
}

static void test_fed_forest_policies_adwin_and_error_paths(void) {
    /* NULL / default guards */
    ht_fed_forest_default_client_config(NULL, 1U, NULL);
    ht_fed_forest_client_cfg_t ccfg;
    ht_fed_forest_default_client_config(&ccfg, 1U, NULL); /* exercises default tree_params branch */
    ht_fed_forest_default_server_config(NULL, HT_TASK_CLASSIFICATION);
    HT_ASSERT_EQ(ht_fed_forest_client_init(NULL, &ccfg), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_forest_client_set_adwin_pool(NULL, NULL), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_forest_server_init(NULL, NULL), HT_ERR_INVALID_ARG);
    ht_fed_forest_server_advance_round(NULL);

    static ht_fed_forest_client_t cli;
    static ht_adwin_pool_t adwin_pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&adwin_pool), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&cli, &ccfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_set_adwin_pool(&cli, &adwin_pool), HT_OK);

    /* Predict/learn NULL guards */
    ht_float_t probas[HT_MAX_CLASSES];
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    HT_ASSERT_EQ(ht_fed_forest_client_predict_proba(NULL, x, probas), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_forest_client_predict(NULL, x), 0U);
    HT_ASSERT_EQ(ht_fed_forest_client_learn_one(NULL, x, 0U, HT_CONST(1.0)), HT_ERR_INVALID_ARG);
    HT_ASSERT_FLOAT_EQ(ht_fed_forest_client_predict_reg(NULL, x), HT_CONST(0.0));
    HT_ASSERT_EQ(ht_fed_forest_client_learn_one_reg(NULL, x, HT_CONST(1.0), HT_CONST(1.0)), HT_ERR_INVALID_ARG);

    /* Export/import/ingest error paths */
    static uint8_t wire[32768];
    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(NULL, wire, sizeof(wire), &w), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(&cli, wire, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(NULL, wire, 0U), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&cli, wire, 4U), HT_ERR_CORRUPT_IMAGE);

    /* Server policies: ALL and ROUND_ROBIN + table saturation beyond HT_FED_SERVER_MAX_TREES */
    static ht_fed_forest_server_t srv;
    ht_fed_forest_server_cfg_t scfg;
    ht_fed_forest_default_server_config(&scfg, HT_TASK_CLASSIFICATION);
    scfg.agg_policy = HT_FED_FOREST_AGG_ROUND_ROBIN;
    scfg.max_forest_size = 2U;
    HT_ASSERT_EQ(ht_fed_forest_server_init(&srv, &scfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(NULL, wire, 0U), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, 4U), HT_ERR_CORRUPT_IMAGE);

    for (uint16_t cid = 1U; cid <= (uint16_t)(HT_FED_SERVER_MAX_TREES + 2U); ++cid) {
        cli.client_id = cid;
        x[0].num = HT_FROM_UINT(cid);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one(&cli, x, (uint8_t)(cid % 2U), HT_CONST(1.0)), HT_OK);
        HT_ASSERT_EQ(ht_fed_forest_client_export_update(&cli, wire, sizeof(wire), &w), HT_OK);
        HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_OK);
    }
    HT_ASSERT_EQ(srv.n_entries, HT_FED_SERVER_MAX_TREES);

    /* Round-robin bundle build, import, and post-import classification learning (updates peer faded error) */
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 1U, true, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&cli, wire, w), HT_OK);
    HT_ASSERT_EQ(cli.n_peers, 2U);
    HT_ASSERT_EQ(ht_fed_forest_client_learn_one(&cli, x, 0U, HT_CONST(1.0)), HT_OK);
    ht_fed_forest_server_advance_round(&srv);

    /* Policy ALL */
    srv.cfg.agg_policy = HT_FED_FOREST_AGG_ALL;
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 1U, false, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&cli, wire, w), HT_OK);
    HT_ASSERT_EQ(cli.n_peers, 2U);
}

#ifndef HT_CLASSIF_ONLY
static void test_fed_forest_high_amplitude_regression(void) {
    /* Reproduces Appliances/NYC Taxi high-amplitude Fed-Forest regression in Q16.16 */
    static ht_fed_forest_client_t c0, c1;
    static ht_fed_forest_server_t srv;
    static uint8_t wire[32768];

    ht_config_params_t tp;
    ht_default_reg_params(&tp, 2);
    tp.grace_period = 25U;
    tp.delta = HT_CONST(1e-2);
    tp.leaf_prediction = HT_LEAF_MEAN;

    ht_fed_forest_client_cfg_t cfg0, cfg1;
    ht_fed_forest_default_client_config(&cfg0, 1U, &tp);
    ht_fed_forest_default_client_config(&cfg1, 2U, &tp);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&c0, &cfg0), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&c1, &cfg1), HT_OK);

    ht_fed_forest_server_cfg_t scfg;
    ht_fed_forest_default_server_config(&scfg, HT_TASK_REGRESSION);
    scfg.agg_policy = HT_FED_FOREST_AGG_TOP_M;
    HT_ASSERT_EQ(ht_fed_forest_server_init(&srv, &scfg), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    for (int i = 0; i < 120; ++i) {
        x[0].num = HT_FROM_INT(i % 10);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one_reg(&c0, x, HT_CONST(300.0), HT_CONST(1.0)), HT_OK);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one_reg(&c1, x, HT_CONST(3200.0), HT_CONST(1.0)), HT_OK);
    }

    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(&c1, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 1U, true, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&c0, wire, w), HT_OK);

    for (int i = 0; i < 30; ++i) {
        x[0].num = HT_FROM_INT(i % 10);
        HT_ASSERT_EQ(ht_fed_forest_client_learn_one_reg(&c0, x, HT_CONST(300.0), HT_CONST(1.0)), HT_OK);
    }
    ht_float_t pred = ht_fed_forest_client_predict_reg(&c0, x);
    HT_ASSERT(HT_ISFINITE(pred));
    HT_ASSERT(pred > HT_CONST(200.0) && pred < HT_CONST(600.0));
}
#endif

static void test_fed_forest_wire_corruption_and_server_clamps(void) {
    static ht_fed_forest_client_t cli;
    static ht_fed_forest_server_t srv;
    static uint8_t wire[32768];

    ht_config_params_t tp;
    ht_default_params(&tp, 2U, 2U);
    ht_fed_forest_client_cfg_t ccfg;
    ht_fed_forest_default_client_config(&ccfg, 7U, &tp);
    HT_ASSERT_EQ(ht_fed_forest_client_init(&cli, &ccfg), HT_OK);

    /* Server clamping max_forest_size == 0 and > HT_FED_FOREST_MAX_TREES */
    ht_fed_forest_server_cfg_t scfg;
    ht_fed_forest_default_server_config(&scfg, HT_TASK_CLASSIFICATION);
    scfg.max_forest_size = 0U;
    HT_ASSERT_EQ(ht_fed_forest_server_init(&srv, &scfg), HT_OK);
    HT_ASSERT_EQ(srv.cfg.max_forest_size, (uint8_t)HT_FED_FOREST_MAX_TREES);

    /* Export query-size (NULL) + ingest twice for SAME client_id (exercises in-place slot update) */
    size_t req = 0, w = 0;
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(&cli, NULL, 0U, &req), HT_OK);
    HT_ASSERT(req > sizeof(ht_fed_header_t));
    HT_ASSERT_EQ(ht_fed_forest_client_export_update(&cli, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(w, req);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(srv.n_entries, 1U);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(srv.n_entries, 1U);

    /* Build forest query-size (NULL), buffer-too-small, and NULL server guard */
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(NULL, 7U, false, wire, sizeof(wire), &w), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 7U, false, NULL, 0U, &req), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 7U, false, wire, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);

    /* Valid HFED frame with wrong msg_type / task mismatch / truncated payload */
    uint8_t junk[8] = {0};
    uint16_t cls_flags = ht_fed_make_flags(HT_TASK_CLASSIFICATION);
    uint16_t reg_flags = ht_fed_make_flags(HT_TASK_REGRESSION);

    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_VFDT_SPLIT_PROPOSAL, cls_flags, 7U, 1U, 0U, junk, 8U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&cli, wire, w), HT_ERR_CORRUPT_IMAGE);

    /* Task mismatch (reg frame into classif server/client) */
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_FOREST_CLIENT_TREE, reg_flags, 7U, 1U, 0U, junk, 8U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_FOREST_SERVER_BUNDLE, reg_flags, 7U, 1U, 0U, junk, 8U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&cli, wire, w), HT_ERR_CORRUPT_IMAGE);

    /* Truncated bundle header / invalid n_trees inside bundle */
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_FOREST_SERVER_BUNDLE, cls_flags, 7U, 1U, 0U, junk, 2U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&cli, wire, w), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_FOREST_CLIENT_TREE, cls_flags, 7U, 1U, 0U, junk, 2U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_ERR_CORRUPT_IMAGE);

    /* TOP_M tie-breaking (identical reported_error, differing train_weight) & max_forest_size > MAX clamp */
    scfg.max_forest_size = (uint8_t)(HT_FED_FOREST_MAX_TREES + 4U);
    scfg.agg_policy = HT_FED_FOREST_AGG_TOP_M;
    HT_ASSERT_EQ(ht_fed_forest_server_init(&srv, &scfg), HT_OK);
    HT_ASSERT_EQ(srv.cfg.max_forest_size, (uint8_t)HT_FED_FOREST_MAX_TREES);
    for (uint16_t cid = 1U; cid <= 3U; ++cid) {
        cli.client_id = cid;
        cli.member_faded_error[0] = HT_CONST(0.1);
        cli.local_tree.train_weight_seen = HT_FROM_UINT(cid * 10U);
        HT_ASSERT_EQ(ht_fed_forest_client_export_update(&cli, wire, sizeof(wire), &w), HT_OK);
        HT_ASSERT_EQ(ht_fed_forest_server_ingest(&srv, wire, w), HT_OK);
    }
    HT_ASSERT_EQ(ht_fed_forest_server_build_forest(&srv, 99U, false, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_forest_client_import_forest(&cli, wire, w), HT_OK);
    HT_ASSERT_EQ(cli.n_peers, 3U);
}

void test_fed_forest_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1, 2 & 3: Modo 1 Floresta Federada (ht_fed_forest)");
    HT_RUN_TEST(test_fed_proto_framing_and_crc32);
    HT_RUN_TEST(test_fed_forest_server_policies_and_exclude_self);
    HT_RUN_TEST(test_fed_forest_non_iid_collaboration_gain);
    HT_RUN_TEST(test_fed_forest_regression_and_drift_downweighting);
    HT_RUN_TEST(test_fed_proto_corrupt_frames_and_fixed_flag);
    HT_RUN_TEST(test_fed_forest_policies_adwin_and_error_paths);
#ifndef HT_CLASSIF_ONLY
    HT_RUN_TEST(test_fed_forest_high_amplitude_regression);
#endif
    HT_RUN_TEST(test_fed_forest_wire_corruption_and_server_clamps);
}
