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
}

void test_fed_forest_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1, 2 & 3: Modo 1 Floresta Federada (ht_fed_forest)");
    HT_RUN_TEST(test_fed_proto_framing_and_crc32);
    HT_RUN_TEST(test_fed_forest_server_policies_and_exclude_self);
    HT_RUN_TEST(test_fed_forest_non_iid_collaboration_gain);
    HT_RUN_TEST(test_fed_forest_regression_and_drift_downweighting);
}
