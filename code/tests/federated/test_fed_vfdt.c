/**
 * @file test_fed_vfdt.c
 * @brief Level 1, 2 & 3 tests for Mode 2 Fed-VFDT (Coordinated Splitting,
 *        Exact Sufficient Statistics Merge, Straggler Recovery, and Drift Reset).
 */

#include "test_harness.h"
#include "ht_fed_vfdt.h"
#include <string.h>

static void test_fed_vfdt_coordinated_consensus_modes(void) {
    static ht_fed_vfdt_client_t clients[3];
    static ht_fed_vfdt_server_t server;
    static uint8_t wire[32768];

    const ht_fed_vfdt_consensus_t modes[3] = {
        HT_FED_CONSENSUS_QUORUM,
        HT_FED_CONSENSUS_MAJORITY,
        HT_FED_CONSENSUS_MERIT
    };

    for (int m = 0; m < 3; ++m) {
        ht_config_params_t tp;
        ht_default_params(&tp, 2, 2);
        tp.grace_period = 20U;
        tp.delta = (ht_float_t)1e-2;
        tp.leaf_prediction = HT_LEAF_MC;

        ht_fed_vfdt_server_cfg_t scfg;
        ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_COORDINATED, &tp);
        scfg.consensus = modes[m];
        scfg.min_quorum = 2U;
        HT_ASSERT_EQ(ht_fed_vfdt_server_init(&server, &scfg), HT_OK);

        for (int k = 0; k < 3; ++k) {
            ht_fed_vfdt_client_cfg_t ccfg;
            ht_fed_vfdt_default_client_config(&ccfg, (uint16_t)(k + 1), HT_FED_VFDT_COORDINATED, &tp);
            HT_ASSERT_EQ(ht_fed_vfdt_client_init(&clients[k], &ccfg), HT_OK);
        }

        ht_value_t x[HT_MAX_FEATURES];
        memset(x, 0, sizeof(x));

        /* Train each client on cleanly separable data (feature 0 separates class 0 vs 1) */
        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < 120; ++i) {
                uint8_t y = (uint8_t)(i % 2);
                x[0].num = (y == 0U)
                    ? (ht_float_t)(1.0 + (i % 5) * 0.1)
                    : (ht_float_t)(10.0 + (i % 5) * 0.1);
                x[1].num = (ht_float_t)((i + k) % 7);
                HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&clients[k], x, y, (ht_float_t)1.0), HT_OK);
            }
            /* Splits must remain deferred locally */
            HT_ASSERT_EQ(clients[k].tree.n_decision_nodes, 0U);
        }

        /* Coordinated round */
        HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&server), HT_OK);
        for (int k = 0; k < 3; ++k) {
            size_t w = 0;
            HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&clients[k], wire, sizeof(wire), &w), HT_OK);
            HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&server, wire, w), HT_OK);
        }

        uint16_t cmds = 0;
        HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&server, &cmds), HT_OK);
        HT_ASSERT(cmds >= 1U);
        HT_ASSERT_EQ(server.topology_version, 1U);
        HT_ASSERT_EQ(server.global_tree.n_decision_nodes, 1U);
        HT_ASSERT_EQ(server.global_tree.memory.node_pool[0].feature_idx, 0U);

        /* Broadcast to all 3 clients and verify 100% identical synchronized topology */
        for (int k = 0; k < 3; ++k) {
            size_t w = 0;
            HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&server, clients[k].client_id,
                                                          clients[k].topology_version,
                                                          wire, sizeof(wire), &w), HT_OK);
            HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&clients[k], wire, w), HT_OK);
            HT_ASSERT_EQ(clients[k].topology_version, 1U);
            HT_ASSERT_EQ(clients[k].tree.n_decision_nodes, 1U);
            HT_ASSERT_EQ(clients[k].tree.memory.node_pool[0].feature_idx, 0U);
            HT_ASSERT_FLOAT_EQ(clients[k].tree.memory.node_pool[0].split_val.threshold,
                               server.global_tree.memory.node_pool[0].split_val.threshold);
        }
    }
}

static void test_fed_vfdt_stats_merge_extreme_non_iid(void) {
    /* Extreme Non-IID test:
     * Client 0 ONLY sees class 0 (x0 ~ 1.0) -> 100% pure locally, CANNOT split alone.
     * Client 1 ONLY sees class 1 (x0 ~ 10.0) -> 100% pure locally, CANNOT split alone.
     * Exact Stats-Merge combines both observers on the server and unlocks the global split! */
    static ht_fed_vfdt_client_t c0, c1;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_params(&tp, 2, 2);
    tp.grace_period = 50U;
    tp.delta = (ht_float_t)1e-2;
    tp.leaf_prediction = HT_LEAF_MC;

    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_STATS_MERGE, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);

    ht_fed_vfdt_client_cfg_t cfg0, cfg1;
    ht_fed_vfdt_default_client_config(&cfg0, 1U, HT_FED_VFDT_STATS_MERGE, &tp);
    ht_fed_vfdt_default_client_config(&cfg1, 2U, HT_FED_VFDT_STATS_MERGE, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&c0, &cfg0), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&c1, &cfg1), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 80; ++i) {
        x[0].num = (ht_float_t)(1.0 + (i % 5) * 0.1);
        x[1].num = (ht_float_t)(i % 6);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c0, x, 0U, (ht_float_t)1.0), HT_OK);

        x[0].num = (ht_float_t)(10.0 + (i % 5) * 0.1);
        x[1].num = (ht_float_t)(i % 6);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c1, x, 1U, (ht_float_t)1.0), HT_OK);
    }

    /* Confirm neither client can split locally on its own */
    bool c0_ready = false, c1_ready = false;
    HT_ASSERT_EQ(ht_tree_evaluate_leaf_split(&c0.tree, 0U, NULL, NULL, &c0_ready), HT_OK);
    HT_ASSERT_EQ(ht_tree_evaluate_leaf_split(&c1.tree, 0U, NULL, NULL, &c1_ready), HT_OK);
    HT_ASSERT(!c0_ready);
    HT_ASSERT(!c1_ready);

    /* Stats-Merge round on server */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c0, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c1, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);

    uint16_t cmds = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT_EQ(srv.topology_version, 1U);
    HT_ASSERT_EQ(srv.global_tree.memory.node_pool[0].feature_idx, 0U);

    /* Apply global split to both clients -> both now predict BOTH classes accurately! */
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, c0.topology_version, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c0, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 2U, c1.topology_version, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c1, wire, w), HT_OK);

    x[0].num = (ht_float_t)1.2;
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c0, x), 0U);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c1, x), 0U);
    x[0].num = (ht_float_t)10.2;
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c0, x), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c1, x), 1U);
}

static void test_fed_vfdt_stats_merge_regression(void) {
    static ht_fed_vfdt_client_t c0, c1;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_reg_params(&tp, 2);
    tp.grace_period = 40U;
    tp.delta = (ht_float_t)1e-2;
    tp.leaf_prediction = HT_LEAF_MEAN;

    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_STATS_MERGE, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);

    ht_fed_vfdt_client_cfg_t cfg0, cfg1;
    ht_fed_vfdt_default_client_config(&cfg0, 1U, HT_FED_VFDT_STATS_MERGE, &tp);
    ht_fed_vfdt_default_client_config(&cfg1, 2U, HT_FED_VFDT_STATS_MERGE, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&c0, &cfg0), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&c1, &cfg1), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 80; ++i) {
        x[0].num = (ht_float_t)(1.0 + (i % 5) * 0.1);
        x[1].num = (ht_float_t)(i % 4);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one_reg(&c0, x, (ht_float_t)5.0, (ht_float_t)1.0), HT_OK);

        x[0].num = (ht_float_t)(10.0 + (i % 5) * 0.1);
        x[1].num = (ht_float_t)(i % 4);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one_reg(&c1, x, (ht_float_t)50.0, (ht_float_t)1.0), HT_OK);
    }

    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c0, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c1, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);

    uint16_t cmds = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);

    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, c0.topology_version, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c0, wire, w), HT_OK);

    x[0].num = (ht_float_t)1.2;
    HT_ASSERT_FLOAT_NEAR(ht_fed_vfdt_client_predict_reg(&c0, x), (ht_float_t)5.0, (ht_float_t)2.0);
    x[0].num = (ht_float_t)10.2;
    HT_ASSERT_FLOAT_NEAR(ht_fed_vfdt_client_predict_reg(&c0, x), (ht_float_t)50.0, (ht_float_t)2.0);
}

static void test_fed_vfdt_straggler_catchup_snapshot_and_drift_reset(void) {
    static ht_fed_vfdt_client_t c_active, c_straggler;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_params(&tp, 2, 2);
    tp.grace_period = 20U;
    tp.delta = (ht_float_t)1e-2;
    tp.leaf_prediction = HT_LEAF_MC;

    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_COORDINATED, &tp);
    scfg.min_quorum = 1U;
    scfg.min_drift_quorum = 1U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);

    ht_fed_vfdt_client_cfg_t cfg_a, cfg_s;
    ht_fed_vfdt_default_client_config(&cfg_a, 1U, HT_FED_VFDT_COORDINATED, &tp);
    ht_fed_vfdt_default_client_config(&cfg_s, 2U, HT_FED_VFDT_COORDINATED, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&c_active, &cfg_a), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&c_straggler, &cfg_s), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Active client triggers split (version 0 -> 1) while c_straggler is offline */
    for (int i = 0; i < 100; ++i) {
        uint8_t y = (uint8_t)(i % 2);
        x[0].num = (y == 0U) ? (ht_float_t)(1.0 + (i % 5) * 0.1) : (ht_float_t)(10.0 + (i % 5) * 0.1);
        x[1].num = (ht_float_t)(i % 7);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c_active, x, y, (ht_float_t)1.0), HT_OK);
    }
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c_active, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    uint16_t cmds = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT_EQ(srv.topology_version, 1U);

    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, c_active.topology_version, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c_active, wire, w), HT_OK);

    /* Straggler (at version 0) sends stale uplink -> rejected with HT_ERR_STALE_VERSION */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c_straggler, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_ERR_STALE_VERSION);

    /* Server exports incremental split_log catch-up -> straggler reaches version 1 */
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 2U, c_straggler.topology_version, wire, sizeof(wire), &w), HT_OK);
    ht_fed_header_t hdr;
    HT_ASSERT_EQ(ht_fed_parse_frame(wire, w, &hdr, NULL), HT_OK);
    HT_ASSERT_EQ(hdr.msg_type, HT_FED_MSG_VFDT_SPLIT_BROADCAST);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c_straggler, wire, w), HT_OK);
    HT_ASSERT_EQ(c_straggler.topology_version, 1U);
    HT_ASSERT_EQ(c_straggler.tree.n_decision_nodes, 1U);

    /* Force FULL_SNAPSHOT fallback by simulating split_log wraparound ahead of client */
    srv.topology_version = HT_FED_SPLIT_LOG_CAP + 5U;
    srv.split_log_head_version = 6U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 2U, c_straggler.topology_version, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_parse_frame(wire, w, &hdr, NULL), HT_OK);
    HT_ASSERT_EQ(hdr.msg_type, HT_FED_MSG_VFDT_FULL_SNAPSHOT);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c_straggler, wire, w), HT_OK);
    HT_ASSERT_EQ(c_straggler.topology_version, srv.topology_version);
    HT_ASSERT_EQ(c_straggler.tree.n_decision_nodes, 1U);
    HT_ASSERT_EQ(c_straggler.tree.n_active_leaves, 2U);

    /* Coordinated drift reset on root */
    c_active.topology_version = srv.topology_version;
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    ht_fed_vfdt_client_flag_drift(&c_active, c_active.tree.root_idx);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c_active, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT_EQ(srv.global_tree.n_decision_nodes, 0U);
    HT_ASSERT_EQ(srv.global_tree.n_active_leaves, 1U);

    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 2U, c_straggler.topology_version, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c_straggler, wire, w), HT_OK);
    HT_ASSERT_EQ(c_straggler.tree.n_decision_nodes, 0U);
    HT_ASSERT_EQ(c_straggler.tree.n_active_leaves, 1U);
}

void test_fed_vfdt_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1, 2 & 3: Modo 2 Fed-VFDT Global (ht_fed_vfdt)");
    HT_RUN_TEST(test_fed_vfdt_coordinated_consensus_modes);
    HT_RUN_TEST(test_fed_vfdt_stats_merge_extreme_non_iid);
    HT_RUN_TEST(test_fed_vfdt_stats_merge_regression);
    HT_RUN_TEST(test_fed_vfdt_straggler_catchup_snapshot_and_drift_reset);
}
