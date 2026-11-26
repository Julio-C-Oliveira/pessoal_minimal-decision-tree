/**
 * @file fed_simulator.c
 * @brief Multi-client Federated Learning simulator & benchmark for kansho (C11).
 *
 * Compares 5 strategies across IID, Non-IID, Continuous Regression, and Concept Drift:
 *   1. Local-Only (isolated clients)
 *   2. Centralized Oracle (single tree on all data)
 *   3. Mode 1: Fed-Forest (1 local tree + M compact inference peer trees)
 *   4. Mode 2A: Fed-VFDT Coordinated Splitting (Silva, Vinagre & Gama)
 *   5. Mode 2B: Fed-VFDT Exact Sufficient Statistics Merge (Chan/Welford)
 *
 * Measures global accuracy / RMSE, total wire bytes (uplink + downlink), and static SRAM.
 */

#include "ht_tree.h"
#include "ht_drift.h"
#include "ht_compact_tree.h"
#include "ht_fed_forest.h"
#include "ht_fed_vfdt.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#define SIM_CLIENTS      6
#define SIM_ROUNDS       10
#define SIM_PER_ROUND    120
#define SIM_TEST_SAMPLES 600
#define SIM_WIRE_CAP     65536

/* Deterministic Xorshift32 PRNG */
static uint32_t sim_rng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static ht_float_t sim_rand_uniform(uint32_t *state, ht_float_t lo, ht_float_t hi) {
    ht_float_t u = (ht_float_t)(sim_rng_next(state) & 0x00FFFFFFU) / (ht_float_t)16777216.0;
    return lo + u * (hi - lo);
}

/* Sample generators */
static void gen_sea_sample(uint32_t *rng, ht_float_t thresh, ht_value_t x[HT_MAX_FEATURES], uint8_t *y) {
    memset(x, 0, sizeof(ht_value_t) * HT_MAX_FEATURES);
    x[0].num = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0);
    x[1].num = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0);
    x[2].num = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0); /* Irrelevant */
    *y = ((x[0].num + x[1].num) <= thresh) ? 0U : 1U;
}

static void gen_non_iid_4class(uint32_t *rng, int client_id, bool global_eval,
                               ht_value_t x[HT_MAX_FEATURES], uint8_t *y) {
    memset(x, 0, sizeof(ht_value_t) * HT_MAX_FEATURES);
    /* 4-class problem driven by x0 and x1 quadrants:
     * c0: x0 < 5, x1 < 5; c1: x0 < 5, x1 >= 5; c2: x0 >= 5, x1 < 5; c3: x0 >= 5, x1 >= 5.
     * In Non-IID training, each client mainly sees 2 adjacent classes (85% skewed). */
    uint8_t target_c;
    if (global_eval) {
        target_c = (uint8_t)(sim_rng_next(rng) % 4U);
    } else {
        ht_float_t u = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)1.0);
        if (u < (ht_float_t)0.88) {
            uint8_t c_primary = (uint8_t)(client_id % 4);
            uint8_t c_secondary = (uint8_t)((client_id + 1) % 4);
            target_c = (sim_rng_next(rng) & 1U) ? c_primary : c_secondary;
        } else {
            target_c = (uint8_t)(sim_rng_next(rng) % 4U);
        }
    }

    x[0].num = (target_c < 2U)
        ? sim_rand_uniform(rng, (ht_float_t)0.5, (ht_float_t)4.5)
        : sim_rand_uniform(rng, (ht_float_t)5.5, (ht_float_t)9.5);
    x[1].num = ((target_c % 2U) == 0U)
        ? sim_rand_uniform(rng, (ht_float_t)0.5, (ht_float_t)4.5)
        : sim_rand_uniform(rng, (ht_float_t)5.5, (ht_float_t)9.5);
    x[2].num = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0);
    *y = target_c;
}

static void gen_reg_sample(uint32_t *rng, int client_id, bool global_eval,
                           ht_value_t x[HT_MAX_FEATURES], ht_float_t *y) {
    memset(x, 0, sizeof(ht_value_t) * HT_MAX_FEATURES);
    ht_float_t x0;
    if (global_eval) {
        x0 = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)12.0);
    } else {
        /* Overlapping Non-IID x0 ranges across clients */
        ht_float_t lo = (ht_float_t)(client_id * 1.5);
        ht_float_t hi = lo + (ht_float_t)4.5;
        if (hi > (ht_float_t)12.0) hi = (ht_float_t)12.0;
        x0 = sim_rand_uniform(rng, lo, hi);
    }
    ht_float_t x1 = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)5.0);
    ht_float_t noise = sim_rand_uniform(rng, (ht_float_t)-0.5, (ht_float_t)0.5);
    x[0].num = x0;
    x[1].num = x1;
    x[2].num = sim_rand_uniform(rng, (ht_float_t)0.0, (ht_float_t)10.0);
    /* Piecewise continuous surface */
    if (x0 < (ht_float_t)4.0) {
        *y = (ht_float_t)5.0 + (ht_float_t)0.5 * x1 + noise;
    } else if (x0 < (ht_float_t)8.0) {
        *y = (ht_float_t)25.0 + (ht_float_t)0.5 * x1 + noise;
    } else {
        *y = (ht_float_t)55.0 + (ht_float_t)0.5 * x1 + noise;
    }
}

/* Static simulation instances (zero heap allocation) */
static ht_tree_t              g_central;
static ht_tree_t              g_local[SIM_CLIENTS];
static ht_fed_forest_client_t g_ff_clients[SIM_CLIENTS];
static ht_fed_forest_server_t g_ff_server;
static ht_adwin_pool_t        g_ff_adwin[SIM_CLIENTS];
static ht_fed_vfdt_client_t   g_vc_clients[SIM_CLIENTS];
static ht_fed_vfdt_server_t   g_vc_server;
static ht_fed_vfdt_client_t   g_vs_clients[SIM_CLIENTS];
static ht_fed_vfdt_server_t   g_vs_server;
static uint8_t                g_wire[SIM_WIRE_CAP];

static void run_classification_scenario(const char *name, bool non_iid, bool with_drift) {
    uint8_t n_classes = non_iid ? 4U : 2U;
    ht_config_params_t tp;
    ht_default_params(&tp, 3U, n_classes);
    tp.grace_period = 40U;
    tp.delta = (ht_float_t)1e-2;
    tp.leaf_prediction = HT_LEAF_NBA;

    ht_config_params_t tp_hat = tp;
    if (with_drift) {
        tp_hat.enable_hat = true;
    }

    ht_tree_init(&g_central, &tp);
    for (int k = 0; k < SIM_CLIENTS; ++k) {
        ht_tree_init(&g_local[k], &tp);

        ht_fed_forest_client_cfg_t ff_cfg;
        ht_fed_forest_default_client_config(&ff_cfg, (uint16_t)(k + 1), with_drift ? &tp_hat : &tp);
        ff_cfg.voting = HT_VOTE_WEIGHTED;
        ff_cfg.error_decay = with_drift ? (ht_float_t)0.92 : (ht_float_t)0.98;
        ht_fed_forest_client_init(&g_ff_clients[k], &ff_cfg);
        if (with_drift) {
            ht_adwin_pool_init(&g_ff_adwin[k]);
            ht_fed_forest_client_set_adwin_pool(&g_ff_clients[k], &g_ff_adwin[k]);
        }

        ht_fed_vfdt_client_cfg_t vc_cfg;
        ht_fed_vfdt_default_client_config(&vc_cfg, (uint16_t)(k + 1), HT_FED_VFDT_COORDINATED, &tp);
        ht_fed_vfdt_client_init(&g_vc_clients[k], &vc_cfg);

        ht_fed_vfdt_client_cfg_t vs_cfg;
        ht_fed_vfdt_default_client_config(&vs_cfg, (uint16_t)(k + 1), HT_FED_VFDT_STATS_MERGE, &tp);
        ht_fed_vfdt_client_init(&g_vs_clients[k], &vs_cfg);
    }

    ht_fed_forest_server_cfg_t ff_scfg;
    ht_fed_forest_default_server_config(&ff_scfg, HT_TASK_CLASSIFICATION);
    ff_scfg.agg_policy = HT_FED_FOREST_AGG_ALL;
    ff_scfg.max_forest_size = (uint8_t)SIM_CLIENTS;
    ht_fed_forest_server_init(&g_ff_server, &ff_scfg);

    ht_fed_vfdt_server_cfg_t vc_scfg;
    ht_fed_vfdt_default_server_config(&vc_scfg, HT_FED_VFDT_COORDINATED, &tp);
    vc_scfg.consensus = HT_FED_CONSENSUS_MERIT;
    vc_scfg.min_quorum = 2U;
    vc_scfg.min_drift_quorum = 3U;
    ht_fed_vfdt_server_init(&g_vc_server, &vc_scfg);

    ht_fed_vfdt_server_cfg_t vs_scfg;
    ht_fed_vfdt_default_server_config(&vs_scfg, HT_FED_VFDT_STATS_MERGE, &tp);
    vs_scfg.min_drift_quorum = 3U;
    ht_fed_vfdt_server_init(&g_vs_server, &vs_scfg);

    size_t ff_bytes = 0U, vc_bytes = 0U, vs_bytes = 0U;
    uint32_t rng = non_iid ? 0xC0FFEE22U : (with_drift ? 0xD121F701U : 0x5EA12345U);

    for (int r = 0; r < SIM_ROUNDS; ++r) {
        ht_float_t sea_thresh = (with_drift && r >= (SIM_ROUNDS / 2)) ? (ht_float_t)13.0 : (ht_float_t)7.0;
        bool drift_trigger_round = (with_drift && r == (SIM_ROUNDS / 2));

        for (int k = 0; k < SIM_CLIENTS; ++k) {
            ht_float_t round_err = (ht_float_t)0.0;
            for (int i = 0; i < SIM_PER_ROUND; ++i) {
                ht_value_t x[HT_MAX_FEATURES];
                uint8_t y = 0U;
                if (non_iid) {
                    gen_non_iid_4class(&rng, k, false, x, &y);
                } else {
                    gen_sea_sample(&rng, sea_thresh, x, &y);
                }
                if (ht_fed_vfdt_client_predict(&g_vs_clients[k], x) != y) {
                    round_err += (ht_float_t)1.0;
                }
                ht_tree_learn_one(&g_central, x, y, (ht_float_t)1.0);
                ht_tree_learn_one(&g_local[k], x, y, (ht_float_t)1.0);
                ht_fed_forest_client_learn_one(&g_ff_clients[k], x, y, (ht_float_t)1.0);
                ht_fed_vfdt_client_learn_one(&g_vc_clients[k], x, y, (ht_float_t)1.0);
                ht_fed_vfdt_client_learn_one(&g_vs_clients[k], x, y, (ht_float_t)1.0);
            }
            if (drift_trigger_round || (with_drift && (round_err / (ht_float_t)SIM_PER_ROUND) > (ht_float_t)0.35)) {
                ht_fed_vfdt_client_flag_drift(&g_vc_clients[k], g_vc_clients[k].tree.root_idx);
                ht_fed_vfdt_client_flag_drift(&g_vs_clients[k], g_vs_clients[k].tree.root_idx);
            }
        }

        /* Round exchange: Mode 1 Fed-Forest */
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_forest_client_export_update(&g_ff_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                ff_bytes += w;
                ht_fed_forest_server_ingest(&g_ff_server, g_wire, w);
            }
        }
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_forest_server_build_forest(&g_ff_server, g_ff_clients[k].client_id, true,
                                                  g_wire, sizeof(g_wire), &w) == HT_OK) {
                ff_bytes += w;
                ht_fed_forest_client_import_forest(&g_ff_clients[k], g_wire, w);
            }
        }
        ht_fed_forest_server_advance_round(&g_ff_server);

        /* Round exchange: Mode 2A Fed-VFDT Coordinated */
        ht_fed_vfdt_server_begin_round(&g_vc_server);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_client_export_update(&g_vc_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                vc_bytes += w;
                ht_fed_vfdt_server_ingest(&g_vc_server, g_wire, w);
            }
        }
        ht_fed_vfdt_server_end_round(&g_vc_server, NULL);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_server_export_update(&g_vc_server, g_vc_clients[k].client_id,
                                                 g_vc_clients[k].topology_version,
                                                 g_wire, sizeof(g_wire), &w) == HT_OK) {
                vc_bytes += w;
                ht_fed_vfdt_client_apply_update(&g_vc_clients[k], g_wire, w);
            }
        }

        /* Round exchange: Mode 2B Fed-VFDT Stats-Merge */
        ht_fed_vfdt_server_begin_round(&g_vs_server);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_client_export_update(&g_vs_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                vs_bytes += w;
                ht_fed_vfdt_server_ingest(&g_vs_server, g_wire, w);
            }
        }
        ht_fed_vfdt_server_end_round(&g_vs_server, NULL);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_server_export_update(&g_vs_server, g_vs_clients[k].client_id,
                                                 g_vs_clients[k].topology_version,
                                                 g_wire, sizeof(g_wire), &w) == HT_OK) {
                vs_bytes += w;
                ht_fed_vfdt_client_apply_update(&g_vs_clients[k], g_wire, w);
            }
        }
    }

    /* Global holdout evaluation */
    uint32_t eval_rng = 0x99887766U;
    ht_float_t eval_thresh = with_drift ? (ht_float_t)13.0 : (ht_float_t)7.0;
    int ok_central = 0, ok_local = 0, ok_ff = 0, ok_vc = 0, ok_vs = 0;

    for (int i = 0; i < SIM_TEST_SAMPLES; ++i) {
        ht_value_t x[HT_MAX_FEATURES];
        uint8_t y = 0U;
        if (non_iid) {
            gen_non_iid_4class(&eval_rng, 0, true, x, &y);
        } else {
            gen_sea_sample(&eval_rng, eval_thresh, x, &y);
        }
        if (ht_tree_predict(&g_central, x) == y) ok_central++;
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            if (ht_tree_predict(&g_local[k], x) == y) ok_local++;
            if (ht_fed_forest_client_predict(&g_ff_clients[k], x) == y) ok_ff++;
            if (ht_fed_vfdt_client_predict(&g_vc_clients[k], x) == y) ok_vc++;
            if (ht_fed_vfdt_client_predict(&g_vs_clients[k], x) == y) ok_vs++;
        }
    }

    double acc_central = 100.0 * (double)ok_central / (double)SIM_TEST_SAMPLES;
    double acc_local   = 100.0 * (double)ok_local / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS);
    double acc_ff      = 100.0 * (double)ok_ff / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS);
    double acc_vc      = 100.0 * (double)ok_vc / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS);
    double acc_vs      = 100.0 * (double)ok_vs / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS);

    printf("\n### Scenario: %s (%d clients x %d rounds x %d samples)\n",
           name, SIM_CLIENTS, SIM_ROUNDS, SIM_PER_ROUND);
    printf("| Strategy | Global Holdout Acc | Total Wire Traffic | Bytes/Client/Round | Splits |\n");
    printf("| :--- | :---: | :---: | :---: | :---: |\n");
    printf("| Local-Only (Baseline)      | %6.2f%% | %7.2f KB | %6.1f B | %u (c0) |\n",
           acc_local, 0.0, 0.0, g_local[0].n_decision_nodes);
    printf("| Centralized Oracle         | %6.2f%% |     N/A    |    N/A   | %u |\n",
           acc_central, g_central.n_decision_nodes);
    printf("| Mode 1: Fed-Forest         | %6.2f%% | %7.2f KB | %6.1f B | %u (local) + %u peers |\n",
           acc_ff, (double)ff_bytes / 1024.0, (double)ff_bytes / (double)(SIM_CLIENTS * SIM_ROUNDS),
           g_ff_clients[0].local_tree.n_decision_nodes, g_ff_clients[0].n_peers);
    printf("| Mode 2A: Fed-VFDT (Coord)  | %6.2f%% | %7.2f KB | %6.1f B | %u (v%u) |\n",
           acc_vc, (double)vc_bytes / 1024.0, (double)vc_bytes / (double)(SIM_CLIENTS * SIM_ROUNDS),
           g_vc_server.global_tree.n_decision_nodes, g_vc_server.topology_version);
    printf("| Mode 2B: Fed-VFDT (Merge)  | %6.2f%% | %7.2f KB | %6.1f B | %u (v%u) |\n",
           acc_vs, (double)vs_bytes / 1024.0, (double)vs_bytes / (double)(SIM_CLIENTS * SIM_ROUNDS),
           g_vs_server.global_tree.n_decision_nodes, g_vs_server.topology_version);
}

static void run_regression_scenario(void) {
    ht_config_params_t tp;
    ht_default_reg_params(&tp, 3U);
    tp.grace_period = 40U;
    tp.delta = (ht_float_t)1e-2;
    tp.leaf_prediction = HT_LEAF_MEAN;

    ht_tree_init(&g_central, &tp);
    for (int k = 0; k < SIM_CLIENTS; ++k) {
        ht_tree_init(&g_local[k], &tp);

        ht_fed_forest_client_cfg_t ff_cfg;
        ht_fed_forest_default_client_config(&ff_cfg, (uint16_t)(k + 1), &tp);
        ff_cfg.voting = HT_VOTE_UNIFORM;
        ht_fed_forest_client_init(&g_ff_clients[k], &ff_cfg);

        ht_fed_vfdt_client_cfg_t vc_cfg;
        ht_fed_vfdt_default_client_config(&vc_cfg, (uint16_t)(k + 1), HT_FED_VFDT_COORDINATED, &tp);
        ht_fed_vfdt_client_init(&g_vc_clients[k], &vc_cfg);

        ht_fed_vfdt_client_cfg_t vs_cfg;
        ht_fed_vfdt_default_client_config(&vs_cfg, (uint16_t)(k + 1), HT_FED_VFDT_STATS_MERGE, &tp);
        ht_fed_vfdt_client_init(&g_vs_clients[k], &vs_cfg);
    }

    ht_fed_forest_server_cfg_t ff_scfg;
    ht_fed_forest_default_server_config(&ff_scfg, HT_TASK_REGRESSION);
    ff_scfg.agg_policy = HT_FED_FOREST_AGG_ALL;
    ff_scfg.max_forest_size = (uint8_t)SIM_CLIENTS;
    ht_fed_forest_server_init(&g_ff_server, &ff_scfg);

    ht_fed_vfdt_server_cfg_t vc_scfg;
    ht_fed_vfdt_default_server_config(&vc_scfg, HT_FED_VFDT_COORDINATED, &tp);
    vc_scfg.consensus = HT_FED_CONSENSUS_MERIT;
    vc_scfg.min_quorum = 2U;
    ht_fed_vfdt_server_init(&g_vc_server, &vc_scfg);

    ht_fed_vfdt_server_cfg_t vs_scfg;
    ht_fed_vfdt_default_server_config(&vs_scfg, HT_FED_VFDT_STATS_MERGE, &tp);
    ht_fed_vfdt_server_init(&g_vs_server, &vs_scfg);

    size_t ff_bytes = 0U, vc_bytes = 0U, vs_bytes = 0U;
    uint32_t rng = 0x5EEDBEEFU;

    for (int r = 0; r < SIM_ROUNDS; ++r) {
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            for (int i = 0; i < SIM_PER_ROUND; ++i) {
                ht_value_t x[HT_MAX_FEATURES];
                ht_float_t y = (ht_float_t)0.0;
                gen_reg_sample(&rng, k, false, x, &y);
                ht_tree_learn_one_reg(&g_central, x, y, (ht_float_t)1.0);
                ht_tree_learn_one_reg(&g_local[k], x, y, (ht_float_t)1.0);
                ht_fed_forest_client_learn_one_reg(&g_ff_clients[k], x, y, (ht_float_t)1.0);
                ht_fed_vfdt_client_learn_one_reg(&g_vc_clients[k], x, y, (ht_float_t)1.0);
                ht_fed_vfdt_client_learn_one_reg(&g_vs_clients[k], x, y, (ht_float_t)1.0);
            }
        }

        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_forest_client_export_update(&g_ff_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                ff_bytes += w;
                ht_fed_forest_server_ingest(&g_ff_server, g_wire, w);
            }
        }
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_forest_server_build_forest(&g_ff_server, g_ff_clients[k].client_id, true,
                                                  g_wire, sizeof(g_wire), &w) == HT_OK) {
                ff_bytes += w;
                ht_fed_forest_client_import_forest(&g_ff_clients[k], g_wire, w);
            }
        }
        ht_fed_forest_server_advance_round(&g_ff_server);

        ht_fed_vfdt_server_begin_round(&g_vc_server);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_client_export_update(&g_vc_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                vc_bytes += w;
                ht_fed_vfdt_server_ingest(&g_vc_server, g_wire, w);
            }
        }
        ht_fed_vfdt_server_end_round(&g_vc_server, NULL);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_server_export_update(&g_vc_server, g_vc_clients[k].client_id,
                                                 g_vc_clients[k].topology_version,
                                                 g_wire, sizeof(g_wire), &w) == HT_OK) {
                vc_bytes += w;
                ht_fed_vfdt_client_apply_update(&g_vc_clients[k], g_wire, w);
            }
        }

        ht_fed_vfdt_server_begin_round(&g_vs_server);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_client_export_update(&g_vs_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                vs_bytes += w;
                ht_fed_vfdt_server_ingest(&g_vs_server, g_wire, w);
            }
        }
        ht_fed_vfdt_server_end_round(&g_vs_server, NULL);
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            size_t w = 0U;
            if (ht_fed_vfdt_server_export_update(&g_vs_server, g_vs_clients[k].client_id,
                                                 g_vs_clients[k].topology_version,
                                                 g_wire, sizeof(g_wire), &w) == HT_OK) {
                vs_bytes += w;
                ht_fed_vfdt_client_apply_update(&g_vs_clients[k], g_wire, w);
            }
        }
    }

    uint32_t eval_rng = 0x11223344U;
    double se_central = 0.0, se_local = 0.0, se_ff = 0.0, se_vc = 0.0, se_vs = 0.0;

    for (int i = 0; i < SIM_TEST_SAMPLES; ++i) {
        ht_value_t x[HT_MAX_FEATURES];
        ht_float_t y = (ht_float_t)0.0;
        gen_reg_sample(&eval_rng, 0, true, x, &y);

        double dc = (double)ht_tree_predict_reg(&g_central, x) - (double)y;
        se_central += dc * dc;
        for (int k = 0; k < SIM_CLIENTS; ++k) {
            double dl = (double)ht_tree_predict_reg(&g_local[k], x) - (double)y;
            double df = (double)ht_fed_forest_client_predict_reg(&g_ff_clients[k], x) - (double)y;
            double dv = (double)ht_fed_vfdt_client_predict_reg(&g_vc_clients[k], x) - (double)y;
            double ds = (double)ht_fed_vfdt_client_predict_reg(&g_vs_clients[k], x) - (double)y;
            se_local += dl * dl;
            se_ff += df * df;
            se_vc += dv * dv;
            se_vs += ds * ds;
        }
    }

    double rmse_central = sqrt(se_central / (double)SIM_TEST_SAMPLES);
    double rmse_local   = sqrt(se_local / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS));
    double rmse_ff      = sqrt(se_ff / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS));
    double rmse_vc      = sqrt(se_vc / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS));
    double rmse_vs      = sqrt(se_vs / (double)(SIM_TEST_SAMPLES * SIM_CLIENTS));

    printf("\n### Scenario: Piecewise Continuous Regression (Non-IID Ranges, %d clients x %d rounds)\n",
           SIM_CLIENTS, SIM_ROUNDS);
    printf("| Strategy | Global Holdout RMSE | Total Wire Traffic | Bytes/Client/Round | Splits |\n");
    printf("| :--- | :---: | :---: | :---: | :---: |\n");
    printf("| Local-Only (Baseline)      | %7.3f | %7.2f KB | %6.1f B | %u (c0) |\n",
           rmse_local, 0.0, 0.0, g_local[0].n_decision_nodes);
    printf("| Centralized Oracle         | %7.3f |     N/A    |    N/A   | %u |\n",
           rmse_central, g_central.n_decision_nodes);
    printf("| Mode 1: Fed-Forest         | %7.3f | %7.2f KB | %6.1f B | %u (local) + %u peers |\n",
           rmse_ff, (double)ff_bytes / 1024.0, (double)ff_bytes / (double)(SIM_CLIENTS * SIM_ROUNDS),
           g_ff_clients[0].local_tree.n_decision_nodes, g_ff_clients[0].n_peers);
    printf("| Mode 2A: Fed-VFDT (Coord)  | %7.3f | %7.2f KB | %6.1f B | %u (v%u) |\n",
           rmse_vc, (double)vc_bytes / 1024.0, (double)vc_bytes / (double)(SIM_CLIENTS * SIM_ROUNDS),
           g_vc_server.global_tree.n_decision_nodes, g_vc_server.topology_version);
    printf("| Mode 2B: Fed-VFDT (Merge)  | %7.3f | %7.2f KB | %6.1f B | %u (v%u) |\n",
           rmse_vs, (double)vs_bytes / 1024.0, (double)vs_bytes / (double)(SIM_CLIENTS * SIM_ROUNDS),
           g_vs_server.global_tree.n_decision_nodes, g_vs_server.topology_version);
}

int main(void) {
    printf("===========================================================================\n");
    printf(" kansho Federated Learning Simulator (Pure C11, Zero Dynamic Allocation)\n");
    printf("===========================================================================\n");
    printf("Static SRAM Footprint (Current Compile Profile):\n");
    printf("  - ht_compact_tree_t       : %6zu B (%.2f KB) [max %u nodes]\n",
           sizeof(ht_compact_tree_t), (double)sizeof(ht_compact_tree_t) / 1024.0, HT_COMPACT_MAX_NODES);
    printf("  - ht_tree_t (1 local)     : %6zu B (%.2f KB) [max %u nodes, %u active leaves]\n",
           sizeof(ht_tree_t), (double)sizeof(ht_tree_t) / 1024.0, HT_MAX_NODES, HT_MAX_ACTIVE_LEAVES);
    printf("  - ht_fed_forest_client_t  : %6zu B (%.2f KB) [1 local + %u compact peers]\n",
           sizeof(ht_fed_forest_client_t), (double)sizeof(ht_fed_forest_client_t) / 1024.0, HT_FED_FOREST_MAX_TREES);
    printf("  - ht_fed_forest_server_t  : %6zu B (%.2f KB) [capacity %u trees]\n",
           sizeof(ht_fed_forest_server_t), (double)sizeof(ht_fed_forest_server_t) / 1024.0, HT_FED_SERVER_MAX_TREES);
    printf("  - ht_fed_vfdt_client_t    : %6zu B (%.2f KB)\n",
           sizeof(ht_fed_vfdt_client_t), (double)sizeof(ht_fed_vfdt_client_t) / 1024.0);
    printf("  - ht_fed_vfdt_server_t    : %6zu B (%.2f KB) [split_log=%u]\n",
           sizeof(ht_fed_vfdt_server_t), (double)sizeof(ht_fed_vfdt_server_t) / 1024.0, HT_FED_SPLIT_LOG_CAP);

    run_classification_scenario("SEA Classification (IID)", false, false);
    run_classification_scenario("4-Class Quadrant (Non-IID Skewed)", true, false);
    run_classification_scenario("SEA Abrupt Concept Drift", false, true);
    run_regression_scenario();

    return 0;
}
