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

    const ht_fed_vfdt_consensus_t modes[4] = {
        HT_FED_CONSENSUS_QUORUM,
        HT_FED_CONSENSUS_MAJORITY,
        HT_FED_CONSENSUS_MERIT,
        HT_FED_CONSENSUS_WEIGHTED_MERIT
    };

    for (int m = 0; m < 4; ++m) {
        ht_config_params_t tp;
        ht_default_params(&tp, 2, 2);
        tp.grace_period = 20U;
        tp.delta = HT_CONST(1e-2);
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
                ht_float_t jitter = HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.1));
                x[0].num = (y == 0U)
                    ? HT_ADD(HT_CONST(1.0), jitter)
                    : HT_ADD(HT_CONST(10.0), jitter);
                x[1].num = HT_FROM_INT((i + k) % 7);
                HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&clients[k], x, y, HT_CONST(1.0)), HT_OK);
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
    tp.delta = HT_CONST(1e-2);
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
        ht_float_t jitter = HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.1));
        x[0].num = HT_ADD(HT_CONST(1.0), jitter);
        x[1].num = HT_FROM_INT(i % 6);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c0, x, 0U, HT_CONST(1.0)), HT_OK);

        x[0].num = HT_ADD(HT_CONST(10.0), jitter);
        x[1].num = HT_FROM_INT(i % 6);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c1, x, 1U, HT_CONST(1.0)), HT_OK);
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

    x[0].num = HT_CONST(1.2);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c0, x), 0U);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c1, x), 0U);
    x[0].num = HT_CONST(10.2);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c0, x), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(&c1, x), 1U);
}

static void test_fed_vfdt_stats_merge_regression(void) {
#ifdef HT_CLASSIF_ONLY
    static ht_fed_vfdt_server_t srv;
    ht_config_params_t tp;
    ht_default_reg_params(&tp, 2);
    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_STATS_MERGE, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_ERR_INVALID_ARG);
#else
    static ht_fed_vfdt_client_t c0, c1;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_reg_params(&tp, 2);
    tp.grace_period = 40U;
    tp.delta = HT_CONST(1e-2);
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
        ht_float_t jitter = HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.1));
        x[0].num = HT_ADD(HT_CONST(1.0), jitter);
        x[1].num = HT_FROM_INT(i % 4);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one_reg(&c0, x, HT_CONST(5.0), HT_CONST(1.0)), HT_OK);

        x[0].num = HT_ADD(HT_CONST(10.0), jitter);
        x[1].num = HT_FROM_INT(i % 4);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one_reg(&c1, x, HT_CONST(50.0), HT_CONST(1.0)), HT_OK);
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

    x[0].num = HT_CONST(1.2);
    HT_ASSERT_FLOAT_NEAR(ht_fed_vfdt_client_predict_reg(&c0, x), HT_CONST(5.0), HT_CONST(2.0));
    x[0].num = HT_CONST(10.2);
    HT_ASSERT_FLOAT_NEAR(ht_fed_vfdt_client_predict_reg(&c0, x), HT_CONST(50.0), HT_CONST(2.0));
#endif
}

static void test_fed_vfdt_straggler_catchup_snapshot_and_drift_reset(void) {
    static ht_fed_vfdt_client_t c_active, c_straggler;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_params(&tp, 2, 2);
    tp.grace_period = 20U;
    tp.delta = HT_CONST(1e-2);
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
        ht_float_t jitter = HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.1));
        x[0].num = (y == 0U) ? HT_ADD(HT_CONST(1.0), jitter) : HT_ADD(HT_CONST(10.0), jitter);
        x[1].num = HT_FROM_INT(i % 7);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c_active, x, y, HT_CONST(1.0)), HT_OK);
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

static void helper_push_proposal(ht_fed_vfdt_server_t *srv,
                                 uint16_t leaf_idx,
                                 uint8_t feature_idx,
                                 ht_float_t threshold,
                                 ht_float_t merit,
                                 ht_float_t total_weight,
                                 uint8_t locally_ready) {
    ht_fed_split_proposal_t p;
    memset(&p, 0, sizeof(p));
    ht_split_candidate_init(&p.best);
    p.best.feature_idx = feature_idx;
    p.best.is_categorical = 0U;
    p.best.split_val.threshold = threshold;
    p.best.merit = merit;
    p.best.lhs_dist[0] = HT_MUL(total_weight, HT_CONST(0.5));
    p.best.rhs_dist[1] = HT_MUL(total_weight, HT_CONST(0.5));
    p.total_weight = total_weight;
    p.leaf_idx = leaf_idx;
    p.depth = (leaf_idx == 0U) ? 0U : 1U;
    p.locally_ready = locally_ready;
    srv->round_proposals[srv->n_round_proposals++] = p;
}

static void test_fed_vfdt_paper_exact_consensus_and_thresholds(void) {
    static ht_fed_vfdt_server_t srv;
    ht_config_params_t tp;
    ht_default_params(&tp, 4, 2);

    /* 1. QUORUM: per-feature quorum must REJECT split when 4 clients split 2 vs 2 (min_quorum=3) */
    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_COORDINATED, &tp);
    scfg.consensus = HT_FED_CONSENSUS_QUORUM;
    scfg.min_quorum = 3U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(5.0), HT_CONST(0.8), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(5.2), HT_CONST(0.7), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 1U, HT_CONST(3.0), HT_CONST(0.9), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 1U, HT_CONST(3.1), HT_CONST(0.85), HT_CONST(100.0), 1U);
    uint16_t cmds = 0U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 0U);

    /* Add 3rd vote for f1 (higher merit sum) + 3rd vote for f0 -> f1 wins by higher sum(merit)
     * and threshold must equal exact max-merit proposal (3.0, merit 0.90) */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(5.0), HT_CONST(0.6), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(5.2), HT_CONST(0.6), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(5.4), HT_CONST(0.6), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 1U, HT_CONST(3.0), HT_CONST(0.90), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 1U, HT_CONST(3.6), HT_CONST(0.70), HT_CONST(100.0), 1U);
    helper_push_proposal(&srv, 0U, 1U, HT_CONST(3.9), HT_CONST(0.65), HT_CONST(100.0), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT_EQ(srv.global_tree.memory.node_pool[0].feature_idx, 1U);
    HT_ASSERT_FLOAT_EQ(srv.global_tree.memory.node_pool[0].split_val.threshold, HT_CONST(3.0));

    /* 2. BEST-MERIT (max individual) vs WEIGHTED_MERIT (sum w_k * G_k) + WEIGHTED_MEAN threshold */
    scfg.consensus = HT_FED_CONSENSUS_MERIT;
    scfg.threshold_agg = HT_FED_THRESH_MAX_MERIT;
    scfg.min_quorum = 2U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(4.0), HT_CONST(0.50), HT_CONST(200.0), 1U);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(6.0), HT_CONST(0.50), HT_CONST(200.0), 1U);
    helper_push_proposal(&srv, 0U, 2U, HT_CONST(7.5), HT_CONST(0.92), HT_CONST(30.0), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT_EQ(srv.global_tree.memory.node_pool[0].feature_idx, 2U);
    HT_ASSERT_FLOAT_EQ(srv.global_tree.memory.node_pool[0].split_val.threshold, HT_CONST(7.5));

    /* Same proposals under WEIGHTED_MERIT + WEIGHTED_MEAN -> f0 wins with weighted mean 5.0 */
    scfg.consensus = HT_FED_CONSENSUS_WEIGHTED_MERIT;
    scfg.threshold_agg = HT_FED_THRESH_WEIGHTED_MEAN;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(4.0), HT_CONST(0.50), HT_CONST(200.0), 1U);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(6.0), HT_CONST(0.50), HT_CONST(200.0), 1U);
    helper_push_proposal(&srv, 0U, 2U, HT_CONST(7.5), HT_CONST(0.92), HT_CONST(30.0), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT_EQ(srv.global_tree.memory.node_pool[0].feature_idx, 0U);
    HT_ASSERT_FLOAT_EQ(srv.global_tree.memory.node_pool[0].split_val.threshold, HT_CONST(5.0));

    /* 3. MAJORITY-VOTE plurality (< 50% 3-way vote) + require_locally_ready filter */
    scfg.consensus = HT_FED_CONSENSUS_MAJORITY;
    scfg.threshold_agg = HT_FED_THRESH_MAX_MERIT;
    scfg.min_quorum = 2U;
    scfg.require_locally_ready = true;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);

    /* Unready proposals (locally_ready = 0) must be ignored */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(2.0), HT_CONST(0.80), HT_CONST(50.0), 0U);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(2.0), HT_CONST(0.80), HT_CONST(50.0), 0U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 0U);

    /* 3-way plurality: f0 (2 votes, max merit 0.60), f1 (2 votes, max merit 0.85), f2 (1 vote) -> f1 wins! */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(2.0), HT_CONST(0.60), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(2.1), HT_CONST(0.55), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, 0U, 1U, HT_CONST(4.5), HT_CONST(0.85), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, 0U, 1U, HT_CONST(4.9), HT_CONST(0.70), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, 0U, 2U, HT_CONST(8.0), HT_CONST(0.95), HT_CONST(50.0), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT_EQ(srv.global_tree.memory.node_pool[0].feature_idx, 1U);
    HT_ASSERT_FLOAT_EQ(srv.global_tree.memory.node_pool[0].split_val.threshold, HT_CONST(4.5));

    /* 4. max_splits_per_round = 1 caps splitting to highest-voted leaf when 2 leaves are ready */
    uint16_t left_leaf = srv.global_tree.memory.node_pool[0].left_child;
    uint16_t right_leaf = srv.global_tree.memory.node_pool[0].right_child;
    srv.cfg.max_splits_per_round = 1U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    /* left_leaf gets 2 votes, right_leaf gets 3 votes -> only right_leaf must split */
    helper_push_proposal(&srv, left_leaf, 0U, HT_CONST(1.0), HT_CONST(0.8), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, left_leaf, 0U, HT_CONST(1.1), HT_CONST(0.7), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, right_leaf, 2U, HT_CONST(6.0), HT_CONST(0.75), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, right_leaf, 2U, HT_CONST(6.2), HT_CONST(0.70), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, right_leaf, 2U, HT_CONST(6.4), HT_CONST(0.65), HT_CONST(50.0), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT(srv.global_tree.memory.node_pool[left_leaf].flags & HT_NODE_IS_LEAF);
    HT_ASSERT(!(srv.global_tree.memory.node_pool[right_leaf].flags & HT_NODE_IS_LEAF));
}

static void test_fed_vfdt_predict_proba_depth_and_error_paths(void) {
    /* NULL / default guards */
    ht_fed_vfdt_default_client_config(NULL, 1U, HT_FED_VFDT_COORDINATED, NULL);
    ht_fed_vfdt_client_cfg_t ccfg;
    ht_fed_vfdt_default_client_config(&ccfg, 1U, HT_FED_VFDT_COORDINATED, NULL);
    ht_fed_vfdt_default_server_config(NULL, HT_FED_VFDT_COORDINATED, NULL);
    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_COORDINATED, NULL);

    HT_ASSERT_EQ(ht_fed_vfdt_client_init(NULL, &ccfg), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(NULL, &scfg), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(NULL), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(NULL, NULL), HT_ERR_INVALID_ARG);

    static ht_fed_vfdt_client_t cli;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&cli, &ccfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    ht_float_t probas[HT_MAX_CLASSES];

    /* Cover ht_fed_vfdt_client_predict_proba (was 0%) + NULL guards */
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict_proba(NULL, x, probas), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict_proba(&cli, x, probas), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict(NULL, x), 0U);
    HT_ASSERT_FLOAT_EQ(ht_fed_vfdt_client_predict_reg(NULL, x), HT_CONST(0.0));
    HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(NULL, x, 0U, HT_CONST(1.0)), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one_reg(NULL, x, HT_CONST(1.0), HT_CONST(1.0)), HT_ERR_INVALID_ARG);

    /* Export / apply / ingest corrupt and buffer-too-small paths */
    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(NULL, wire, sizeof(wire), &w), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli, wire, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(NULL, wire, 0U), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli, wire, 4U), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(NULL, wire, 0U), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, 4U), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(NULL, 1U, 0U, wire, sizeof(wire), &w), HT_ERR_INVALID_ARG);

    /* Multi-level split depth computation (covers ht_fed_compute_node_depth) */
    srv.cfg.min_quorum = 1U;
    srv.cfg.consensus = HT_FED_CONSENSUS_QUORUM;
    srv.cfg.min_drift_quorum = 2U;

    /* Round 1: split root -> children (depth 1) */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(5.0), HT_CONST(0.9), HT_CONST(50.0), 1U);
    uint16_t cmds = 0U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);

    /* Round 2: split right child (depth 1 -> depth 2) */
    uint16_t rc = srv.global_tree.memory.node_pool[0].right_child;
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    helper_push_proposal(&srv, rc, 1U, HT_CONST(3.0), HT_CONST(0.8), HT_CONST(50.0), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT(srv.global_tree.max_depth_seen >= 2U);

    /* Drift vote accumulation across 2 clients (reaches min_drift_quorum = 2) */
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 0U, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli, wire, w), HT_OK);

    static ht_fed_vfdt_client_t cli2;
    ht_fed_vfdt_default_client_config(&ccfg, 2U, HT_FED_VFDT_COORDINATED, NULL);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&cli2, &ccfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli2, wire, w), HT_OK);

    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    ht_fed_vfdt_client_flag_drift(&cli, 0U);
    ht_fed_vfdt_client_flag_drift(&cli2, 0U);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli2, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT(cmds >= 1U);
}

static void test_fed_vfdt_stats_merge_hist_and_categorical(void) {
    /* Covers ht_hist_merge and ht_categorical_merge branches inside ht_fed_vfdt.c */
    static ht_fed_vfdt_client_t c0, c1;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_params(&tp, 2, 2);
    tp.feature_types[0] = HT_FEAT_NUMERIC;
    tp.feature_types[1] = HT_FEAT_CATEGORICAL;
    tp.numeric_obs_type = HT_OBS_HISTOGRAM;
    tp.grace_period = 30U;
    tp.delta = HT_CONST(1e-2);
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
    for (int i = 0; i < 60; ++i) {
        x[0].num = HT_ADD(HT_CONST(1.0), HT_MUL(HT_FROM_INT(i % 4), HT_CONST(0.2)));
        x[1].cat = i % 2; /* noisy categorical feature so f0 strictly wins without tie */
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c0, x, 0U, HT_CONST(1.0)), HT_OK);

        x[0].num = HT_ADD(HT_CONST(10.0), HT_MUL(HT_FROM_INT(i % 4), HT_CONST(0.2)));
        x[1].cat = (i + 1) % 2;
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&c1, x, 1U, HT_CONST(1.0)), HT_OK);
    }

    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c0, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c1, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);

    uint16_t cmds = 0U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);

    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 0U, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c0, wire, w), HT_OK);

    ht_float_t probas[HT_MAX_CLASSES];
    x[0].num = HT_CONST(10.2);
    x[1].cat = 1;
    HT_ASSERT_EQ(ht_fed_vfdt_client_predict_proba(&c0, x, probas), HT_OK);
    HT_ASSERT(probas[1] > HT_CONST(0.7));
}

#ifndef HT_CLASSIF_ONLY
static void test_fed_vfdt_stats_merge_high_amplitude_rescaling_and_cat(void) {
    /* Reproduces NYC Taxi IID STATS-MERGE Block-Rescaling stall + covers ht_reg_categorical_merge */
    static ht_fed_vfdt_client_t c0, c1;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_reg_params(&tp, 2);
    tp.feature_types[0] = HT_FEAT_NUMERIC;
    tp.feature_types[1] = HT_FEAT_CATEGORICAL;
    tp.grace_period = 150U;
    ht_params_set_delta_pow10(&tp, 3U);
    tp.leaf_prediction = HT_LEAF_ADAPTIVE;

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

    for (int i = 0; i < 200; ++i) {
        x[0].num = HT_ADD(HT_CONST(1.0), HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.2)));
        x[1].cat = i % 3; /* noisy categorical feature so f0 strictly wins without tie */
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one_reg(&c0, x, HT_ADD(HT_CONST(300.0), HT_FROM_INT(i % 30)), HT_CONST(1.0)), HT_OK);

        x[0].num = HT_ADD(HT_CONST(20.0), HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.2)));
        x[1].cat = (i + 1) % 3;
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one_reg(&c1, x, HT_ADD(HT_CONST(3000.0), HT_FROM_INT(i % 40)), HT_CONST(1.0)), HT_OK);
    }

    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    size_t w = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c0, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&c1, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);

    uint16_t cmds = 0U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);

    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 0U, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&c0, wire, w), HT_OK);

    x[0].num = HT_CONST(1.2); x[1].cat = 0;
    ht_float_t p_lo = ht_fed_vfdt_client_predict_reg(&c0, x);
    x[0].num = HT_CONST(20.2); x[1].cat = 1;
    ht_float_t p_hi = ht_fed_vfdt_client_predict_reg(&c0, x);
    HT_ASSERT(HT_ISFINITE(p_lo) && HT_ISFINITE(p_hi));
    HT_ASSERT(p_hi > HT_ADD(p_lo, HT_CONST(1000.0)));
}
#endif

static void test_fed_vfdt_coordinated_cat_wire_corruption_and_snapshots(void) {
    static ht_fed_vfdt_client_t cli;
    static ht_fed_vfdt_server_t srv;
    static uint8_t wire[65536];

    ht_config_params_t tp;
    ht_default_params(&tp, 2U, 2U);
    tp.feature_types[0] = HT_FEAT_NUMERIC;
    tp.feature_types[1] = HT_FEAT_CATEGORICAL;
    tp.grace_period = 20U;

    /* Zero quorum clamping in server init */
    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, HT_FED_VFDT_COORDINATED, &tp);
    scfg.min_quorum = 0U;
    scfg.min_drift_quorum = 0U;
    scfg.consensus = HT_FED_CONSENSUS_WEIGHTED_MERIT;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &scfg), HT_OK);
    HT_ASSERT_EQ(srv.cfg.min_quorum, 1U);
    HT_ASSERT_EQ(srv.cfg.min_drift_quorum, 1U);

    ht_fed_vfdt_client_cfg_t ccfg;
    ht_fed_vfdt_default_client_config(&ccfg, 1U, HT_FED_VFDT_COORDINATED, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&cli, &ccfg), HT_OK);

    /* 1. Categorical coordinated proposals -> exercises categorical consensus branch (L939, L972) */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    ht_fed_split_proposal_t pc1, pc2;
    memset(&pc1, 0, sizeof(pc1));
    pc1.leaf_idx = 0U; pc1.best.feature_idx = 1U; pc1.best.is_categorical = true;
    pc1.best.split_val.cat_val = 2; pc1.best.merit = HT_CONST(0.9);
    pc1.total_weight = HT_CONST(40.0); pc1.locally_ready = 1U;
    pc2 = pc1;
    pc2.best.split_val.cat_val = 2; pc2.best.merit = HT_CONST(0.8); pc2.total_weight = HT_CONST(60.0);
    srv.round_proposals[srv.n_round_proposals++] = pc1;
    srv.round_proposals[srv.n_round_proposals++] = pc2;
    uint16_t cmds = 0U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);
    HT_ASSERT((srv.global_tree.memory.node_pool[0].flags & HT_NODE_SPLIT_CAT) != 0U);
    HT_ASSERT_EQ(srv.global_tree.memory.node_pool[0].split_val.cat_val, 2);

    /* 2. Query-size (NULL) & buffer-too-small for client/server export + snapshot fallback */
    size_t req = 0, w = 0;
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli, NULL, 0U, &req), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 0U, NULL, 0U, &req), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 0U, wire, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);

    /* Future client_topology_version > server version forces FULL_SNAPSHOT fallback */
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 999U, NULL, 0U, &req), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 999U, wire, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 999U, wire, sizeof(wire), &w), HT_OK);
    ht_fed_header_t hdr;
    const uint8_t *pl = NULL;
    HT_ASSERT_EQ(ht_fed_parse_frame(wire, w, &hdr, &pl), HT_OK);
    HT_ASSERT_EQ(hdr.msg_type, (uint16_t)HT_FED_MSG_VFDT_FULL_SNAPSHOT);

    /* Apply broadcast twice: first updates v0->v1, second hits already-applied skip */
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 0U, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli, wire, w), HT_OK);
    HT_ASSERT_EQ(cli.topology_version, 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli, wire, w), HT_OK);

    /* 3. Valid HFED frames with wrong msg_type / truncated payload -> HT_ERR_CORRUPT_IMAGE */
    uint8_t junk[4] = {0};
    uint16_t cls_flags = ht_fed_make_flags(HT_TASK_CLASSIFICATION);
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_FOREST_CLIENT_TREE, cls_flags, 1U, 1U, 1U, junk, 4U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli, wire, w), HT_ERR_CORRUPT_IMAGE);

    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_VFDT_SPLIT_BROADCAST, cls_flags, 1U, 1U, 1U, junk, 1U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli, wire, w), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_VFDT_SPLIT_PROPOSAL, cls_flags, 1U, 1U, 1U, junk, 1U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_ERR_CORRUPT_IMAGE);

    /* 4. STATS-MERGE round where leaf does NOT reach grace_period -> preserves saved class_counts (L1106) */
    ht_fed_vfdt_server_cfg_t mcfg;
    ht_fed_vfdt_default_server_config(&mcfg, HT_FED_VFDT_STATS_MERGE, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &mcfg), HT_OK);
    ht_fed_vfdt_default_client_config(&ccfg, 1U, HT_FED_VFDT_STATS_MERGE, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&cli, &ccfg), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    for (int i = 0; i < 5; ++i) {
        x[0].num = HT_FROM_INT(i);
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&cli, x, (uint8_t)(i % 2), HT_CONST(1.0)), HT_OK);
    }
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli, NULL, 0U, &req), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli, wire, 8U, &w), HT_ERR_BUFFER_TOO_SMALL);
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 0U);
    /* Stateless STATS-MERGE restores baseline leaf counts when no split occurs (zero double-counting) */
    HT_ASSERT_FLOAT_EQ(srv.global_tree.memory.node_pool[0].class_counts[0], HT_CONST(0.0));

    /* Corrupt LEAF_STATS payload */
    HT_ASSERT_EQ(ht_fed_write_frame(wire, sizeof(wire), HT_FED_MSG_VFDT_LEAF_STATS, cls_flags, 1U, 1U, 0U, junk, 1U, &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_ingest(&srv, wire, w), HT_ERR_CORRUPT_IMAGE);

    /* 5. Multi-level STATS-MERGE split (depth >= 2) + competing categorical consensus +
     *    reverse leaf_idx sorting & tie-breakers (BEST_MERIT / MAJORITY) */
    ht_fed_vfdt_server_cfg_t c2;
    ht_fed_vfdt_default_server_config(&c2, HT_FED_VFDT_COORDINATED, &tp);
    c2.min_quorum = 1U;
    c2.consensus = HT_FED_CONSENSUS_MERIT;
    c2.require_locally_ready = true;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &c2), HT_OK);

    /* Split root -> l1, r1; then split r1 -> l2, r2; then submit proposals in reverse leaf order (r2, l1) */
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    pc1.leaf_idx = 0U; pc1.best.feature_idx = 1U; pc1.best.is_categorical = true;
    pc1.best.split_val.cat_val = 1; pc1.best.merit = HT_CONST(0.7); pc1.locally_ready = 1U;
    pc2 = pc1;
    pc2.best.split_val.cat_val = 2; pc2.best.merit = HT_CONST(0.9); pc2.total_weight = HT_CONST(80.0);
    /* Also add a not-locally-ready proposal to exercise require_locally_ready filter */
    ht_fed_split_proposal_t pc_unready = pc1;
    pc_unready.locally_ready = 0U;
    srv.round_proposals[srv.n_round_proposals++] = pc_unready;
    srv.round_proposals[srv.n_round_proposals++] = pc1;
    srv.round_proposals[srv.n_round_proposals++] = pc2;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);

    uint16_t l1 = srv.global_tree.memory.node_pool[0].left_child;
    uint16_t r1 = srv.global_tree.memory.node_pool[0].right_child;
    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    /* Propose r1 (higher idx) BEFORE l1 (lower idx) to exercise unique_leaves insertion sort */
    helper_push_proposal(&srv, r1, 0U, HT_CONST(4.0), HT_CONST(0.8), HT_CONST(50.0), 1U);
    helper_push_proposal(&srv, l1, 0U, HT_CONST(2.0), HT_CONST(0.85), HT_CONST(50.0), 1U);
    /* Also push a stale proposal for already-split root (non-leaf skip) */
    helper_push_proposal(&srv, 0U, 0U, HT_CONST(1.0), HT_CONST(0.9), HT_CONST(50.0), 1U);
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 2U);

    /* Sync a coordinated client to the multi-level topology and export proposals
     * -> exercises ht_fed_compute_node_depth() DFS traversal (line 315) on non-root leaves */
    ht_fed_vfdt_default_client_config(&ccfg, 3U, HT_FED_VFDT_COORDINATED, &tp);
    HT_ASSERT_EQ(ht_fed_vfdt_client_init(&cli, &ccfg), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 3U, 0U, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_vfdt_client_apply_update(&cli, wire, w), HT_OK);
    for (int i = 0; i < 40; ++i) {
        x[0].num = HT_FROM_INT(i % 10);
        x[1].cat = i % 3;
        HT_ASSERT_EQ(ht_fed_vfdt_client_learn_one(&cli, x, (uint8_t)(i % 2), HT_CONST(1.0)), HT_OK);
    }
    HT_ASSERT_EQ(ht_fed_vfdt_client_export_update(&cli, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT(srv.global_tree.n_decision_nodes >= 3U);

#ifndef HT_CLASSIF_ONLY
    /* 6. Coordinated regression proposals (merges lhs_var/rhs_var in consensus) +
     *    HT_FED_CONSENSUS_MAJORITY tie-breaker & split_log overflow -> FULL_SNAPSHOT */
    ht_config_params_t rtp;
    ht_default_reg_params(&rtp, 2U);
    rtp.grace_period = 20U;
    ht_fed_vfdt_server_cfg_t rcfg;
    ht_fed_vfdt_default_server_config(&rcfg, HT_FED_VFDT_COORDINATED, &rtp);
    rcfg.min_quorum = 2U;
    rcfg.consensus = HT_FED_CONSENSUS_MAJORITY;
    rcfg.threshold_agg = HT_FED_THRESH_WEIGHTED_MEAN;
    HT_ASSERT_EQ(ht_fed_vfdt_server_init(&srv, &rcfg), HT_OK);

    HT_ASSERT_EQ(ht_fed_vfdt_server_begin_round(&srv), HT_OK);
    ht_fed_split_proposal_t rp1, rp2;
    memset(&rp1, 0, sizeof(rp1));
    rp1.leaf_idx = 0U; rp1.best.feature_idx = 0U; rp1.best.split_val.threshold = HT_CONST(4.0);
    rp1.best.merit = HT_CONST(120.0); rp1.total_weight = HT_CONST(50.0); rp1.locally_ready = 1U;
    rp1.best.lhs_var.count = HT_CONST(25.0); rp1.best.lhs_var.mean = HT_CONST(10.0);
    rp1.best.rhs_var.count = HT_CONST(25.0); rp1.best.rhs_var.mean = HT_CONST(90.0);
    rp2 = rp1;
    rp2.best.split_val.threshold = HT_CONST(6.0); rp2.best.merit = HT_CONST(140.0);
    srv.round_proposals[srv.n_round_proposals++] = rp1;
    srv.round_proposals[srv.n_round_proposals++] = rp2;
    HT_ASSERT_EQ(ht_fed_vfdt_server_end_round(&srv, &cmds), HT_OK);
    HT_ASSERT_EQ(cmds, 1U);

    /* Simulate split_log head advancing past client_version -> triggers FULL_SNAPSHOT */
    srv.split_log_head_version = 10U;
    srv.topology_version = 75U;
    HT_ASSERT_EQ(ht_fed_vfdt_server_export_update(&srv, 1U, 2U, wire, sizeof(wire), &w), HT_OK);
    HT_ASSERT_EQ(ht_fed_parse_frame(wire, w, &hdr, &pl), HT_OK);
    HT_ASSERT_EQ(hdr.msg_type, (uint16_t)HT_FED_MSG_VFDT_FULL_SNAPSHOT);
#endif
}

void test_fed_vfdt_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1, 2 & 3: Modo 2 Fed-VFDT Global (ht_fed_vfdt)");
    HT_RUN_TEST(test_fed_vfdt_coordinated_consensus_modes);
    HT_RUN_TEST(test_fed_vfdt_paper_exact_consensus_and_thresholds);
    HT_RUN_TEST(test_fed_vfdt_stats_merge_extreme_non_iid);
    HT_RUN_TEST(test_fed_vfdt_stats_merge_regression);
    HT_RUN_TEST(test_fed_vfdt_straggler_catchup_snapshot_and_drift_reset);
    HT_RUN_TEST(test_fed_vfdt_predict_proba_depth_and_error_paths);
    HT_RUN_TEST(test_fed_vfdt_stats_merge_hist_and_categorical);
#ifndef HT_CLASSIF_ONLY
    HT_RUN_TEST(test_fed_vfdt_stats_merge_high_amplitude_rescaling_and_cat);
#endif
    HT_RUN_TEST(test_fed_vfdt_coordinated_cat_wire_corruption_and_snapshots);
}
