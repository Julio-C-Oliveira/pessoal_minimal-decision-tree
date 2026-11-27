/**
 * @file fed_c_runner.c
 * @brief Pure C11 Multi-Client Federated Learning Benchmark Runner for main_benchmark/.
 *
 * Evaluates 8 strategies on real CSV streams partitioned across K clients:
 *   - local              : K isolated clients (average global holdout)
 *   - central            : 1 centralized oracle tree on full train CSV
 *   - fed_forest_topm    : Mode 1 Fed-Forest (TOP_M + Weighted voting, exclude_self)
 *   - fed_forest_rr      : Mode 1 Fed-Forest (ROUND_ROBIN + Weighted voting, exclude_self)
 *   - fed_vfdt_quorum    : Mode 2A Fed-VFDT Coordinated (QUORUM per-feature, MAX_MERIT)
 *   - fed_vfdt_majority  : Mode 2A Fed-VFDT Coordinated (MAJORITY plurality, MAX_MERIT)
 *   - fed_vfdt_merit     : Mode 2A Fed-VFDT Coordinated (BEST-MERIT max(G_k), MAX_MERIT)
 *   - fed_vfdt_merge     : Mode 2B Fed-VFDT Exact Sufficient Statistics Merge
 *
 * Emits a single clean JSON object to stdout for federated_benchmark.py.
 */

#include "ht_config.h"
#include "ht_tree.h"
#include "ht_prep.h"
#include "ht_compact_tree.h"
#include "ht_fed_forest.h"
#include "ht_fed_vfdt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

static inline ht_float_t fed_from_double(double v) {
#ifdef HT_USE_FIXED_POINT
    if (isnan(v)) return HT_NAN_VAL;
    if (v >= 32767.9999) return INT32_MAX;
    if (v <= -32767.9999) return -INT32_MAX;
    double scaled = v * 65536.0;
    return (ht_float_t)(scaled >= 0.0 ? (scaled + 0.5) : (scaled - 0.5));
#else
    return (ht_float_t)v;
#endif
}

static inline double fed_to_double(ht_float_t v) {
#ifdef HT_USE_FIXED_POINT
    if (HT_ISNAN(v)) return NAN;
    return (double)v / 65536.0;
#else
    return (double)v;
#endif
}

#define MAX_SIM_CLIENTS 16
#define LINE_BUF_SIZE   4096
#define WIRE_BUF_SIZE   524288

typedef enum {
    STRAT_LOCAL = 0,
    STRAT_CENTRAL,
    STRAT_FED_FOREST_TOPM,
    STRAT_FED_FOREST_RR,
    STRAT_FED_VFDT_QUORUM,
    STRAT_FED_VFDT_MAJORITY,
    STRAT_FED_VFDT_MERIT,
    STRAT_FED_VFDT_MERGE
} sim_strategy_t;

typedef struct {
    char     task[32];
    uint8_t  n_features;
    uint8_t  n_classes;
    uint32_t n_samples;
    ht_feature_type_t feature_types[HT_MAX_FEATURES];
} csv_meta_t;

/* Static model pools (zero heap allocation in C11 core & federated modules) */
static ht_tree_t              g_central_tree;
static ht_scaler_t            g_central_scaler;

static ht_tree_t              g_local_trees[MAX_SIM_CLIENTS];
static ht_fed_forest_client_t g_ff_clients[MAX_SIM_CLIENTS];
static ht_fed_forest_server_t g_ff_server;
static ht_fed_vfdt_client_t   g_fv_clients[MAX_SIM_CLIENTS];
static ht_fed_vfdt_server_t   g_fv_server;
static ht_scaler_t            g_client_scalers[MAX_SIM_CLIENTS];

static uint8_t                g_wire[WIRE_BUF_SIZE];

static void init_standard_scaler(ht_scaler_t *sc, uint8_t n_features, const ht_feature_type_t ftypes[HT_MAX_FEATURES]) {
    ht_scaler_init(sc);
    for (uint8_t f = 0U; f < n_features; ++f) {
        if (ftypes[f] == HT_FEAT_NUMERIC) {
            ht_scaler_set_mode(sc, f, HT_SCALER_STANDARD);
        }
    }
}

static void scaler_learn_transform_vec(ht_scaler_t *sc,
                                       uint8_t n_features,
                                       const ht_feature_type_t ftypes[HT_MAX_FEATURES],
                                       const ht_value_t x_in[HT_MAX_FEATURES],
                                       ht_value_t x_out[HT_MAX_FEATURES]) {
    for (uint8_t f = 0U; f < n_features; ++f) {
        if (ftypes[f] == HT_FEAT_NUMERIC) {
            x_out[f].num = ht_scaler_learn_transform_one(sc, f, x_in[f].num, HT_CONST(1.0));
        } else {
            x_out[f] = x_in[f];
        }
    }
}

static void scaler_transform_vec(const ht_scaler_t *sc,
                                 uint8_t n_features,
                                 const ht_feature_type_t ftypes[HT_MAX_FEATURES],
                                 const ht_value_t x_in[HT_MAX_FEATURES],
                                 ht_value_t x_out[HT_MAX_FEATURES]) {
    for (uint8_t f = 0U; f < n_features; ++f) {
        if (ftypes[f] == HT_FEAT_NUMERIC) {
            x_out[f].num = ht_scaler_transform_one(sc, f, x_in[f].num);
        } else {
            x_out[f] = x_in[f];
        }
    }
}

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static bool parse_csv_header(FILE *fp, csv_meta_t *meta) {
    char line[LINE_BUF_SIZE];
    memset(meta, 0, sizeof(*meta));
    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        meta->feature_types[f] = HT_FEAT_NUMERIC;
    }

    if (!fgets(line, sizeof(line), fp)) {
        return false;
    }
    if (line[0] != '#') {
        rewind(fp);
        return false;
    }

    char *tok = strtok(line + 1, " \t\r\n");
    while (tok != NULL) {
        if (strncmp(tok, "task=", 5) == 0) {
            snprintf(meta->task, sizeof(meta->task), "%s", tok + 5);
        } else if (strncmp(tok, "n_features=", 11) == 0) {
            meta->n_features = (uint8_t)atoi(tok + 11);
        } else if (strncmp(tok, "n_classes=", 10) == 0) {
            meta->n_classes = (uint8_t)atoi(tok + 10);
        } else if (strncmp(tok, "n_samples=", 10) == 0) {
            meta->n_samples = (uint32_t)strtoul(tok + 10, NULL, 10);
        } else if (strncmp(tok, "feature_types=", 14) == 0) {
            const char *p = tok + 14;
            uint8_t idx = 0U;
            while (*p != '\0' && idx < HT_MAX_FEATURES) {
                meta->feature_types[idx++] = (*p == '1') ? HT_FEAT_CATEGORICAL : HT_FEAT_NUMERIC;
                while (*p != '\0' && *p != ',') p++;
                if (*p == ',') p++;
            }
        }
        tok = strtok(NULL, " \t\r\n");
    }
    return (meta->n_features > 0U && meta->n_features <= HT_MAX_FEATURES);
}

static bool read_csv_row(FILE *fp,
                         const csv_meta_t *meta,
                         bool is_reg,
                         ht_value_t x_out[HT_MAX_FEATURES],
                         uint8_t *y_class,
                         ht_float_t *y_reg) {
    char line[LINE_BUF_SIZE];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        memset(x_out, 0, sizeof(ht_value_t) * HT_MAX_FEATURES);
        char *p = line;
        for (uint8_t f = 0; f < meta->n_features; ++f) {
            char *endp = NULL;
            if (meta->feature_types[f] == HT_FEAT_CATEGORICAL) {
                long v = strtol(p, &endp, 10);
                x_out[f].cat = (int32_t)v;
            } else {
                double v = strtod(p, &endp);
                x_out[f].num = fed_from_double(v);
            }
            if (endp == p) {
                return false;
            }
            p = endp;
            if (*p == ',') p++;
        }
        char *endp = NULL;
        if (is_reg) {
            double yv = strtod(p, &endp);
            if (endp == p) return false;
            *y_reg = fed_from_double(yv);
        } else {
            long yc = strtol(p, &endp, 10);
            if (endp == p) return false;
            *y_class = (uint8_t)yc;
        }
        return true;
    }
    return false;
}

static double compute_macro_f1(uint64_t conf[HT_MAX_CLASSES][HT_MAX_CLASSES], uint8_t n_classes) {
    double f1_sum = 0.0;
    int valid_classes = 0;
    for (uint8_t c = 0; c < n_classes; ++c) {
        uint64_t tp = conf[c][c];
        uint64_t fp = 0U, fn = 0U;
        for (uint8_t j = 0; j < n_classes; ++j) {
            if (j != c) {
                fp += conf[j][c];
                fn += conf[c][j];
            }
        }
        if (tp + fp + fn == 0U) {
            continue;
        }
        valid_classes++;
        double denom = (double)(2U * tp + fp + fn);
        f1_sum += (denom > 0.0) ? ((2.0 * (double)tp) / denom) : 0.0;
    }
    return (valid_classes > 0) ? (f1_sum / (double)valid_classes) : 0.0;
}

static sim_strategy_t parse_strategy(const char *s) {
    if (strcmp(s, "local") == 0) return STRAT_LOCAL;
    if (strcmp(s, "central") == 0) return STRAT_CENTRAL;
    if (strcmp(s, "fed_forest_topm") == 0) return STRAT_FED_FOREST_TOPM;
    if (strcmp(s, "fed_forest_rr") == 0) return STRAT_FED_FOREST_RR;
    if (strcmp(s, "fed_vfdt_quorum") == 0) return STRAT_FED_VFDT_QUORUM;
    if (strcmp(s, "fed_vfdt_majority") == 0) return STRAT_FED_VFDT_MAJORITY;
    if (strcmp(s, "fed_vfdt_merit") == 0) return STRAT_FED_VFDT_MERIT;
    if (strcmp(s, "fed_vfdt_merge") == 0) return STRAT_FED_VFDT_MERGE;
    return STRAT_LOCAL;
}

int main(int argc, char **argv) {
    const char *clients_dir = NULL;
    const char *train_csv = NULL;
    const char *test_csv = NULL;
    const char *strategy_str = "local";
    const char *criterion_str = "info_gain";
    const char *leaf_str = "nba";
    const char *obs_str = "gaussian";
    const char *prep_scaler = "none";

    int n_clients = 10;
    int n_rounds = 20;
    int grace_period = 100;
    int client_grace = 0;
    int quorum = 3;
    double delta = 1e-3;
    double tau = 0.05;
    bool leaf_normalize = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--clients-dir") == 0 && i + 1 < argc) clients_dir = argv[++i];
        else if (strcmp(argv[i], "--train-csv") == 0 && i + 1 < argc) train_csv = argv[++i];
        else if (strcmp(argv[i], "--test-csv") == 0 && i + 1 < argc) test_csv = argv[++i];
        else if (strcmp(argv[i], "--strategy") == 0 && i + 1 < argc) strategy_str = argv[++i];
        else if (strcmp(argv[i], "--n-clients") == 0 && i + 1 < argc) n_clients = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) n_rounds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--grace") == 0 && i + 1 < argc) grace_period = atoi(argv[++i]);
        else if (strcmp(argv[i], "--client-grace") == 0 && i + 1 < argc) client_grace = atoi(argv[++i]);
        else if (strcmp(argv[i], "--delta") == 0 && i + 1 < argc) delta = atof(argv[++i]);
        else if (strcmp(argv[i], "--tau") == 0 && i + 1 < argc) tau = atof(argv[++i]);
        else if (strcmp(argv[i], "--criterion") == 0 && i + 1 < argc) criterion_str = argv[++i];
        else if (strcmp(argv[i], "--leaf") == 0 && i + 1 < argc) leaf_str = argv[++i];
        else if (strcmp(argv[i], "--obs") == 0 && i + 1 < argc) obs_str = argv[++i];
        else if (strcmp(argv[i], "--prep-scaler") == 0 && i + 1 < argc) prep_scaler = argv[++i];
        else if (strcmp(argv[i], "--leaf-normalize") == 0) leaf_normalize = true;
        else if (strcmp(argv[i], "--quorum") == 0 && i + 1 < argc) quorum = atoi(argv[++i]);
    }

    if (test_csv == NULL || n_clients < 1 || n_clients > MAX_SIM_CLIENTS || n_rounds < 1) {
        fprintf(stderr, "Usage: %s --test-csv <path> [--clients-dir <dir> | --train-csv <path>] --strategy <...>\n", argv[0]);
        return 1;
    }

    sim_strategy_t strat = parse_strategy(strategy_str);
    bool use_scaler = (strcmp(prep_scaler, "standard") == 0);

    /* Inspect test CSV header for task schema */
    FILE *fp_test = fopen(test_csv, "r");
    if (!fp_test) {
        fprintf(stderr, "Failed to open test CSV: %s\n", test_csv);
        return 1;
    }
    csv_meta_t meta;
    if (!parse_csv_header(fp_test, &meta)) {
        fprintf(stderr, "Failed to parse header in %s\n", test_csv);
        fclose(fp_test);
        return 1;
    }
    fclose(fp_test);

    bool is_reg = (strcmp(meta.task, "regression") == 0);
    if (client_grace <= 0) {
        client_grace = grace_period / n_clients;
        if (client_grace < 20) client_grace = 20;
    }

    /* Configure base tree params */
    ht_config_params_t base_params;
    if (is_reg) {
        ht_default_reg_params(&base_params, meta.n_features);
        if (strcmp(leaf_str, "mean") == 0) base_params.leaf_prediction = HT_LEAF_MEAN;
        else if (strcmp(leaf_str, "model") == 0) base_params.leaf_prediction = HT_LEAF_MODEL;
        else base_params.leaf_prediction = HT_LEAF_ADAPTIVE;
        base_params.leaf_sgd_normalize = leaf_normalize;
    } else {
        ht_default_params(&base_params, meta.n_features, meta.n_classes);
        if (strcmp(criterion_str, "gini") == 0) base_params.criterion = HT_CRIT_GINI;
        else base_params.criterion = HT_CRIT_INFO_GAIN;
        if (strcmp(leaf_str, "mc") == 0) base_params.leaf_prediction = HT_LEAF_MC;
        else if (strcmp(leaf_str, "nb") == 0) base_params.leaf_prediction = HT_LEAF_NB;
        else base_params.leaf_prediction = HT_LEAF_NBA;
        if (strcmp(obs_str, "histogram") == 0 || strcmp(obs_str, "hist") == 0) {
            base_params.numeric_obs_type = HT_OBS_HISTOGRAM;
        } else {
            base_params.numeric_obs_type = HT_OBS_GAUSSIAN;
        }
    }
    for (uint8_t f = 0; f < meta.n_features; ++f) {
        base_params.feature_types[f] = meta.feature_types[f];
    }
    base_params.grace_period = (uint16_t)grace_period;
#ifdef HT_USE_FIXED_POINT
    if (delta > 0.0 && delta < 1e-4) {
        int neg_log10 = (int)(-log10(delta) + 0.5);
        if (neg_log10 < 1) neg_log10 = 1;
        if (neg_log10 > 15) neg_log10 = 15;
        ht_params_set_delta_pow10(&base_params, (uint8_t)neg_log10);
    } else {
        base_params.delta = fed_from_double(delta);
        base_params.delta_neg_log10 = 0U;
    }
#else
    base_params.delta = (ht_float_t)delta;
#endif
    base_params.tau = fed_from_double(tau);

    ht_config_params_t client_params = base_params;
    client_params.grace_period = (uint16_t)client_grace;

    double t0 = now_ms();
    size_t total_uplink_bytes = 0U;
    size_t total_downlink_bytes = 0U;
    size_t max_uplink_frame_bytes = 0U;
    size_t max_downlink_frame_bytes = 0U;
    uint64_t preq_total = 0U;
    uint64_t preq_correct = 0U;
    double preq_abs_err_sum = 0.0;

    if (strat == STRAT_CENTRAL) {
        if (!train_csv) {
            fprintf(stderr, "Missing --train-csv for central strategy\n");
            return 1;
        }
        if (ht_tree_init(&g_central_tree, &base_params) != HT_OK) {
            fprintf(stderr, "ht_tree_init failed for central\n");
            return 1;
        }
        if (use_scaler) {
            init_standard_scaler(&g_central_scaler, meta.n_features, meta.feature_types);
        }

        FILE *fp = fopen(train_csv, "r");
        if (!fp) return 1;
        csv_meta_t tmeta;
        parse_csv_header(fp, &tmeta);

        ht_value_t x_raw[HT_MAX_FEATURES], x_in[HT_MAX_FEATURES];
        uint8_t yc = 0U;
        ht_float_t yr = HT_CONST(0.0);
        while (read_csv_row(fp, &meta, is_reg, x_raw, &yc, &yr)) {
            const ht_value_t *xp = x_raw;
            if (use_scaler) {
                scaler_learn_transform_vec(&g_central_scaler, meta.n_features, meta.feature_types, x_raw, x_in);
                xp = x_in;
            }
            if (is_reg) {
                ht_float_t yp = ht_tree_predict_reg(&g_central_tree, xp);
                preq_abs_err_sum += fabs(fed_to_double(yp) - fed_to_double(yr));
                preq_total++;
                ht_tree_learn_one_reg(&g_central_tree, xp, yr, HT_CONST(1.0));
            } else {
                uint8_t yp = ht_tree_predict(&g_central_tree, xp);
                if (yp == yc) preq_correct++;
                preq_total++;
                ht_tree_learn_one(&g_central_tree, xp, yc, HT_CONST(1.0));
            }
        }
        fclose(fp);
    } else {
        if (!clients_dir) {
            fprintf(stderr, "Missing --clients-dir for federated/local strategy\n");
            return 1;
        }

        /* Open per-client CSV files and read per-client sample counts */
        FILE *cfps[MAX_SIM_CLIENTS];
        uint32_t c_total[MAX_SIM_CLIENTS];
        uint32_t c_seen[MAX_SIM_CLIENTS];
        memset(cfps, 0, sizeof(cfps));
        memset(c_total, 0, sizeof(c_total));
        memset(c_seen, 0, sizeof(c_seen));

        for (int k = 0; k < n_clients; ++k) {
            char path[512];
            snprintf(path, sizeof(path), "%s/client_%d.csv", clients_dir, k);
            cfps[k] = fopen(path, "r");
            if (!cfps[k]) {
                fprintf(stderr, "Failed to open %s\n", path);
                return 1;
            }
            csv_meta_t cmeta;
            parse_csv_header(cfps[k], &cmeta);
            c_total[k] = cmeta.n_samples;

            if (use_scaler) {
                init_standard_scaler(&g_client_scalers[k], meta.n_features, meta.feature_types);
            }
        }

        /* Initialize models for selected strategy */
        if (strat == STRAT_LOCAL) {
            for (int k = 0; k < n_clients; ++k) {
                if (ht_tree_init(&g_local_trees[k], &client_params) != HT_OK) return 1;
            }
        } else if (strat == STRAT_FED_FOREST_TOPM || strat == STRAT_FED_FOREST_RR) {
            ht_fed_forest_server_cfg_t scfg;
            ht_fed_forest_default_server_config(&scfg, base_params.task_type);
            scfg.agg_policy = (strat == STRAT_FED_FOREST_TOPM)
                ? HT_FED_FOREST_AGG_TOP_M : HT_FED_FOREST_AGG_ROUND_ROBIN;
            scfg.max_forest_size = (uint8_t)HT_FED_FOREST_MAX_TREES;
            if (ht_fed_forest_server_init(&g_ff_server, &scfg) != HT_OK) return 1;

            for (int k = 0; k < n_clients; ++k) {
                ht_fed_forest_client_cfg_t ccfg;
                ht_fed_forest_default_client_config(&ccfg, (uint16_t)(k + 1), &client_params);
                ccfg.voting = HT_VOTE_WEIGHTED;
                ccfg.error_decay = HT_CONST(0.98);
                if (ht_fed_forest_client_init(&g_ff_clients[k], &ccfg) != HT_OK) return 1;
            }
        } else {
            ht_fed_vfdt_mode_t vmode = (strat == STRAT_FED_VFDT_MERGE)
                ? HT_FED_VFDT_STATS_MERGE : HT_FED_VFDT_COORDINATED;

            ht_fed_vfdt_server_cfg_t scfg;
            ht_fed_vfdt_default_server_config(&scfg, vmode, &base_params);
            scfg.threshold_agg = HT_FED_THRESH_MAX_MERIT;
            scfg.require_locally_ready = true;
            scfg.max_splits_per_round = 0U;
            if (strat == STRAT_FED_VFDT_QUORUM) {
                scfg.consensus = HT_FED_CONSENSUS_QUORUM;
                scfg.min_quorum = (uint16_t)quorum;
            } else if (strat == STRAT_FED_VFDT_MAJORITY) {
                scfg.consensus = HT_FED_CONSENSUS_MAJORITY;
                scfg.min_quorum = 2U;
            } else if (strat == STRAT_FED_VFDT_MERIT) {
                scfg.consensus = HT_FED_CONSENSUS_MERIT;
                scfg.min_quorum = 2U;
            }
            if (ht_fed_vfdt_server_init(&g_fv_server, &scfg) != HT_OK) return 1;

            for (int k = 0; k < n_clients; ++k) {
                ht_fed_vfdt_client_cfg_t ccfg;
                ht_fed_vfdt_default_client_config(&ccfg, (uint16_t)(k + 1), vmode, &client_params);
                if (ht_fed_vfdt_client_init(&g_fv_clients[k], &ccfg) != HT_OK) return 1;
            }
        }

        /* Round-based federated streaming loop */
        ht_value_t x_raw[HT_MAX_FEATURES], x_in[HT_MAX_FEATURES];
        uint8_t yc = 0U;
        ht_float_t yr = HT_CONST(0.0);

        for (int r = 0; r < n_rounds; ++r) {
            for (int k = 0; k < n_clients; ++k) {
                uint32_t target_end = (uint32_t)(((uint64_t)c_total[k] * (uint64_t)(r + 1)) / (uint64_t)n_rounds);
                while (c_seen[k] < target_end) {
                    if (!read_csv_row(cfps[k], &meta, is_reg, x_raw, &yc, &yr)) {
                        break;
                    }
                    c_seen[k]++;
                    const ht_value_t *xp = x_raw;
                    if (use_scaler) {
                        scaler_learn_transform_vec(&g_client_scalers[k], meta.n_features, meta.feature_types, x_raw, x_in);
                        xp = x_in;
                    }

                    if (strat == STRAT_LOCAL) {
                        if (is_reg) {
                            preq_abs_err_sum += fabs(fed_to_double(ht_tree_predict_reg(&g_local_trees[k], xp)) - fed_to_double(yr));
                            preq_total++;
                            ht_tree_learn_one_reg(&g_local_trees[k], xp, yr, HT_CONST(1.0));
                        } else {
                            if (ht_tree_predict(&g_local_trees[k], xp) == yc) preq_correct++;
                            preq_total++;
                            ht_tree_learn_one(&g_local_trees[k], xp, yc, HT_CONST(1.0));
                        }
                    } else if (strat == STRAT_FED_FOREST_TOPM || strat == STRAT_FED_FOREST_RR) {
                        if (is_reg) {
                            preq_abs_err_sum += fabs(fed_to_double(ht_fed_forest_client_predict_reg(&g_ff_clients[k], xp)) - fed_to_double(yr));
                            preq_total++;
                            ht_fed_forest_client_learn_one_reg(&g_ff_clients[k], xp, yr, HT_CONST(1.0));
                        } else {
                            if (ht_fed_forest_client_predict(&g_ff_clients[k], xp) == yc) preq_correct++;
                            preq_total++;
                            ht_fed_forest_client_learn_one(&g_ff_clients[k], xp, yc, HT_CONST(1.0));
                        }
                    } else {
                        if (is_reg) {
                            preq_abs_err_sum += fabs(fed_to_double(ht_fed_vfdt_client_predict_reg(&g_fv_clients[k], xp)) - fed_to_double(yr));
                            preq_total++;
                            ht_fed_vfdt_client_learn_one_reg(&g_fv_clients[k], xp, yr, HT_CONST(1.0));
                        } else {
                            if (ht_fed_vfdt_client_predict(&g_fv_clients[k], xp) == yc) preq_correct++;
                            preq_total++;
                            ht_fed_vfdt_client_learn_one(&g_fv_clients[k], xp, yc, HT_CONST(1.0));
                        }
                    }
                }
            }

            /* Communication round */
            if (strat == STRAT_FED_FOREST_TOPM || strat == STRAT_FED_FOREST_RR) {
                for (int k = 0; k < n_clients; ++k) {
                    size_t w = 0U;
                    if (ht_fed_forest_client_export_update(&g_ff_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                        total_uplink_bytes += w;
                        if (w > max_uplink_frame_bytes) max_uplink_frame_bytes = w;
                        ht_fed_forest_server_ingest(&g_ff_server, g_wire, w);
                    }
                }
                for (int k = 0; k < n_clients; ++k) {
                    size_t w = 0U;
                    if (ht_fed_forest_server_build_forest(&g_ff_server, g_ff_clients[k].client_id, true,
                                                          g_wire, sizeof(g_wire), &w) == HT_OK) {
                        total_downlink_bytes += w;
                        if (w > max_downlink_frame_bytes) max_downlink_frame_bytes = w;
                        ht_fed_forest_client_import_forest(&g_ff_clients[k], g_wire, w);
                    }
                }
                ht_fed_forest_server_advance_round(&g_ff_server);
            } else if (strat != STRAT_LOCAL) {
                ht_fed_vfdt_server_begin_round(&g_fv_server);
                for (int k = 0; k < n_clients; ++k) {
                    size_t w = 0U;
                    if (ht_fed_vfdt_client_export_update(&g_fv_clients[k], g_wire, sizeof(g_wire), &w) == HT_OK) {
                        total_uplink_bytes += w;
                        if (w > max_uplink_frame_bytes) max_uplink_frame_bytes = w;
                        (void)ht_fed_vfdt_server_ingest(&g_fv_server, g_wire, w);
                    }
                }
                ht_fed_vfdt_server_end_round(&g_fv_server, NULL);
                for (int k = 0; k < n_clients; ++k) {
                    size_t w = 0U;
                    if (ht_fed_vfdt_server_export_update(&g_fv_server, g_fv_clients[k].client_id,
                                                         g_fv_clients[k].topology_version,
                                                         g_wire, sizeof(g_wire), &w) == HT_OK) {
                        total_downlink_bytes += w;
                        if (w > max_downlink_frame_bytes) max_downlink_frame_bytes = w;
                        (void)ht_fed_vfdt_client_apply_update(&g_fv_clients[k], g_wire, w);
                    }
                }
            }
        }

        for (int k = 0; k < n_clients; ++k) {
            fclose(cfps[k]);
        }
    }

    double train_ms = now_ms() - t0;

    /* Global Holdout Evaluation on test_csv */
    fp_test = fopen(test_csv, "r");
    if (!fp_test) return 1;
    csv_meta_t test_meta;
    parse_csv_header(fp_test, &test_meta);

    int eval_models = (strat == STRAT_CENTRAL) ? 1 : n_clients;
    uint64_t conf[MAX_SIM_CLIENTS][HT_MAX_CLASSES][HT_MAX_CLASSES];
    memset(conf, 0, sizeof(conf));
    uint64_t eval_correct[MAX_SIM_CLIENTS];
    double eval_ae[MAX_SIM_CLIENTS];
    double eval_se[MAX_SIM_CLIENTS];
    memset(eval_correct, 0, sizeof(eval_correct));
    memset(eval_ae, 0, sizeof(eval_ae));
    memset(eval_se, 0, sizeof(eval_se));

    uint64_t test_n = 0U;
    double y_sum = 0.0;
    double y_sq_sum = 0.0;

    ht_value_t x_raw[HT_MAX_FEATURES], x_in[HT_MAX_FEATURES];
    uint8_t yc = 0U;
    ht_float_t yr = HT_CONST(0.0);

    while (read_csv_row(fp_test, &meta, is_reg, x_raw, &yc, &yr)) {
        test_n++;
        double yr_d = fed_to_double(yr);
        if (is_reg) {
            y_sum += yr_d;
            y_sq_sum += yr_d * yr_d;
        }

        for (int m = 0; m < eval_models; ++m) {
            const ht_value_t *xp = x_raw;
            if (use_scaler) {
                const ht_scaler_t *sc = (strat == STRAT_CENTRAL) ? &g_central_scaler : &g_client_scalers[m];
                scaler_transform_vec(sc, meta.n_features, meta.feature_types, x_raw, x_in);
                xp = x_in;
            }

            if (is_reg) {
                ht_float_t yp = HT_CONST(0.0);
                if (strat == STRAT_CENTRAL) yp = ht_tree_predict_reg(&g_central_tree, xp);
                else if (strat == STRAT_LOCAL) yp = ht_tree_predict_reg(&g_local_trees[m], xp);
                else if (strat == STRAT_FED_FOREST_TOPM || strat == STRAT_FED_FOREST_RR)
                    yp = ht_fed_forest_client_predict_reg(&g_ff_clients[m], xp);
                else yp = ht_fed_vfdt_client_predict_reg(&g_fv_clients[m], xp);

                double err = fed_to_double(yp) - yr_d;
                eval_ae[m] += fabs(err);
                eval_se[m] += err * err;
            } else {
                uint8_t yp = 0U;
                if (strat == STRAT_CENTRAL) yp = ht_tree_predict(&g_central_tree, xp);
                else if (strat == STRAT_LOCAL) yp = ht_tree_predict(&g_local_trees[m], xp);
                else if (strat == STRAT_FED_FOREST_TOPM || strat == STRAT_FED_FOREST_RR)
                    yp = ht_fed_forest_client_predict(&g_ff_clients[m], xp);
                else yp = ht_fed_vfdt_client_predict(&g_fv_clients[m], xp);

                if (yp == yc) eval_correct[m]++;
                if (yc < HT_MAX_CLASSES && yp < HT_MAX_CLASSES) {
                    conf[m][yc][yp]++;
                }
            }
        }
    }
    fclose(fp_test);

    /* Aggregate metrics across clients */
    double holdout_acc = 0.0, holdout_f1 = 0.0;
    double holdout_mae = 0.0, holdout_rmse = 0.0, holdout_r2 = 0.0;
    double preq_acc = (preq_total > 0U) ? (100.0 * (double)preq_correct / (double)preq_total) : 0.0;
    double preq_mae = (preq_total > 0U) ? (preq_abs_err_sum / (double)preq_total) : 0.0;

    if (test_n > 0U) {
        if (is_reg) {
            double y_mean = y_sum / (double)test_n;
            double ss_tot = y_sq_sum - (double)test_n * y_mean * y_mean;
            for (int m = 0; m < eval_models; ++m) {
                double mae_m = eval_ae[m] / (double)test_n;
                double rmse_m = sqrt(eval_se[m] / (double)test_n);
                double r2_m = (ss_tot > 1e-12) ? (1.0 - eval_se[m] / ss_tot) : 0.0;
                holdout_mae += mae_m;
                holdout_rmse += rmse_m;
                holdout_r2 += r2_m;
            }
            holdout_mae /= (double)eval_models;
            holdout_rmse /= (double)eval_models;
            holdout_r2 /= (double)eval_models;
        } else {
            for (int m = 0; m < eval_models; ++m) {
                holdout_acc += 100.0 * (double)eval_correct[m] / (double)test_n;
                holdout_f1 += 100.0 * compute_macro_f1(conf[m], meta.n_classes);
            }
            holdout_acc /= (double)eval_models;
            holdout_f1 /= (double)eval_models;
        }
    }

    typedef struct {
        uint16_t origin_client_id;
        uint16_t reserved;
        uint32_t tree_bytes;
        ht_float_t reported_error;
    } fed_forest_wire_item_t;

    typedef struct {
        uint16_t leaf_idx;
        uint16_t reserved;
        union {
            ht_float_t    class_counts[HT_MAX_CLASSES];
            ht_var_stat_t reg_stats;
        } stats_u;
    } fed_leaf_stats_wire_item_t;

    size_t client_sram = 0U, server_sram = 0U;
    size_t client_scaler_bytes = use_scaler ? sizeof(ht_scaler_t) : 0U;
    size_t client_model_bytes = 0U, client_coord_bytes = 0U, client_comm_bytes = 0U, client_stack_peak_bytes = 0U;
    size_t server_model_bytes = 0U, server_coord_bytes = 0U, server_comm_bytes = 0U, server_stack_peak_bytes = 0U;
    size_t comm_buf_peak_bytes = (max_uplink_frame_bytes > max_downlink_frame_bytes)
        ? max_uplink_frame_bytes : max_downlink_frame_bytes;
    size_t comm_buf_worst_bytes = 0U;
    double avg_nodes = 0.0;
    uint32_t global_splits = 0U;

    if (strat == STRAT_CENTRAL) {
        client_model_bytes = sizeof(ht_tree_t) + client_scaler_bytes;
        client_sram = client_model_bytes;
        avg_nodes = (double)ht_node_count(&g_central_tree.memory);
        global_splits = g_central_tree.n_decision_nodes;
    } else if (strat == STRAT_LOCAL) {
        client_model_bytes = sizeof(ht_tree_t) + client_scaler_bytes;
        client_sram = client_model_bytes;
        for (int k = 0; k < n_clients; ++k) {
            avg_nodes += (double)ht_node_count(&g_local_trees[k].memory);
            global_splits += g_local_trees[k].n_decision_nodes;
        }
        avg_nodes /= (double)n_clients;
        global_splits /= (uint32_t)n_clients;
    } else if (strat == STRAT_FED_FOREST_TOPM || strat == STRAT_FED_FOREST_RR) {
        size_t peer_trees_bytes = (size_t)HT_FED_FOREST_MAX_TREES * sizeof(ht_compact_tree_t);
        size_t srv_trees_bytes = (size_t)HT_FED_SERVER_MAX_TREES * sizeof(ht_compact_tree_t);
        client_model_bytes = sizeof(ht_tree_t) + peer_trees_bytes + client_scaler_bytes;
        client_coord_bytes = sizeof(ht_fed_forest_client_t) - sizeof(ht_tree_t) - peer_trees_bytes;
        client_comm_bytes = comm_buf_peak_bytes;
        client_stack_peak_bytes = sizeof(ht_compact_tree_t);
        client_sram = sizeof(ht_fed_forest_client_t) + client_scaler_bytes;

        server_model_bytes = srv_trees_bytes;
        server_coord_bytes = sizeof(ht_fed_forest_server_t) - srv_trees_bytes;
        server_comm_bytes = comm_buf_peak_bytes;
        server_stack_peak_bytes = sizeof(ht_compact_tree_t);
        server_sram = sizeof(ht_fed_forest_server_t);

        size_t ct_max_wire = sizeof(ht_compact_serial_header_t) +
                             (size_t)HT_COMPACT_MAX_NODES * sizeof(ht_node_t);
        size_t ff_up_worst = sizeof(ht_fed_header_t) + 2U * sizeof(ht_float_t) + ct_max_wire;
        size_t ff_down_worst = sizeof(ht_fed_header_t) + 4U +
                               (size_t)HT_FED_FOREST_MAX_TREES * (sizeof(fed_forest_wire_item_t) + ct_max_wire);
        comm_buf_worst_bytes = (ff_down_worst > ff_up_worst) ? ff_down_worst : ff_up_worst;

        for (int k = 0; k < n_clients; ++k) {
            avg_nodes += (double)ht_node_count(&g_ff_clients[k].local_tree.memory);
            global_splits += g_ff_clients[k].local_tree.n_decision_nodes;
        }
        avg_nodes /= (double)n_clients;
        global_splits /= (uint32_t)n_clients;
    } else {
        client_model_bytes = sizeof(ht_tree_t) + client_scaler_bytes;
        client_coord_bytes = sizeof(ht_fed_vfdt_client_t) - sizeof(ht_tree_t);
        client_comm_bytes = comm_buf_peak_bytes;
        client_sram = sizeof(ht_fed_vfdt_client_t) + client_scaler_bytes;

        server_model_bytes = sizeof(ht_tree_t);
        server_coord_bytes = sizeof(ht_fed_vfdt_server_t) - sizeof(ht_tree_t);
        server_comm_bytes = comm_buf_peak_bytes;
        server_sram = sizeof(ht_fed_vfdt_server_t);

        size_t vf_bcast_worst = sizeof(ht_fed_header_t) + 12U +
                                (size_t)HT_FED_SPLIT_LOG_CAP * sizeof(ht_fed_split_cmd_t);
        size_t vf_snap_worst = sizeof(ht_fed_header_t) + sizeof(ht_serial_header_t) +
                               (size_t)HT_MAX_NODES * (sizeof(uint16_t) + sizeof(ht_node_t));
        size_t vf_down_worst = (vf_bcast_worst > vf_snap_worst) ? vf_bcast_worst : vf_snap_worst;

        if (strat == STRAT_FED_VFDT_MERGE) {
            size_t leaf_stride = (size_t)g_fv_server.global_tree.memory.leaf_stride;
            size_t vf_merge_up_worst = sizeof(ht_fed_header_t) + 8U +
                                       (size_t)HT_MAX_ACTIVE_LEAVES *
                                       (sizeof(fed_leaf_stats_wire_item_t) + leaf_stride);
            comm_buf_worst_bytes = (vf_merge_up_worst > vf_down_worst) ? vf_merge_up_worst : vf_down_worst;
            client_stack_peak_bytes = sizeof(fed_leaf_stats_wire_item_t) + sizeof(ht_fed_header_t);
            server_stack_peak_bytes = sizeof(ht_leaf_stats_t);
        } else {
            size_t vf_prop_worst = sizeof(ht_fed_header_t) + 4U +
                                   (size_t)HT_FED_MAX_PROPOSALS * sizeof(ht_fed_split_proposal_t);
            comm_buf_worst_bytes = (vf_down_worst > vf_prop_worst) ? vf_down_worst : vf_prop_worst;
            client_stack_peak_bytes = (size_t)HT_FED_MAX_PROPOSALS * sizeof(ht_fed_split_proposal_t);
            server_stack_peak_bytes = (size_t)HT_MAX_ACTIVE_LEAVES * sizeof(ht_fed_split_cmd_t);
        }

        avg_nodes = (double)ht_node_count(&g_fv_server.global_tree.memory);
        global_splits = g_fv_server.global_tree.n_decision_nodes;
    }

    size_t client_total_bytes = client_model_bytes + client_coord_bytes + client_comm_bytes;
    size_t server_total_bytes = server_model_bytes + server_coord_bytes + server_comm_bytes;

    size_t total_wire = total_uplink_bytes + total_downlink_bytes;
    double bytes_per_client_round = (double)total_wire / (double)(n_clients * n_rounds);

    printf("{"
           "\"strategy\":\"%s\","
           "\"task\":\"%s\","
           "\"n_clients\":%d,"
           "\"rounds\":%d,"
           "\"holdout_acc\":%.4f,"
           "\"holdout_f1\":%.4f,"
           "\"preq_acc\":%.4f,"
           "\"holdout_mae\":%.4f,"
           "\"holdout_rmse\":%.4f,"
           "\"holdout_r2\":%.4f,"
           "\"preq_mae\":%.4f,"
           "\"avg_nodes\":%.1f,"
           "\"splits\":%u,"
           "\"uplink_bytes\":%zu,"
           "\"downlink_bytes\":%zu,"
           "\"max_uplink_frame_bytes\":%zu,"
           "\"max_downlink_frame_bytes\":%zu,"
           "\"total_wire_kb\":%.2f,"
           "\"bytes_per_client_round\":%.1f,"
           "\"comm_buf_peak_kb\":%.3f,"
           "\"comm_buf_worst_kb\":%.3f,"
           "\"client_sram_kb\":%.2f,"
           "\"server_sram_kb\":%.2f,"
           "\"client_model_kb\":%.3f,"
           "\"client_scaler_kb\":%.3f,"
           "\"client_coord_kb\":%.3f,"
           "\"client_comm_kb\":%.3f,"
           "\"client_total_kb\":%.3f,"
           "\"client_stack_peak_kb\":%.3f,"
           "\"server_model_kb\":%.3f,"
           "\"server_coord_kb\":%.3f,"
           "\"server_comm_kb\":%.3f,"
           "\"server_total_kb\":%.3f,"
           "\"server_stack_peak_kb\":%.3f,"
           "\"train_ms\":%.2f"
           "}\n",
           strategy_str,
           meta.task,
           n_clients,
           n_rounds,
           holdout_acc,
           holdout_f1,
           preq_acc,
           holdout_mae,
           holdout_rmse,
           holdout_r2,
           preq_mae,
           avg_nodes,
           global_splits,
           total_uplink_bytes,
           total_downlink_bytes,
           max_uplink_frame_bytes,
           max_downlink_frame_bytes,
           (double)total_wire / 1024.0,
           bytes_per_client_round,
           (double)comm_buf_peak_bytes / 1024.0,
           (double)comm_buf_worst_bytes / 1024.0,
           (double)client_sram / 1024.0,
           (double)server_sram / 1024.0,
           (double)client_model_bytes / 1024.0,
           (double)client_scaler_bytes / 1024.0,
           (double)client_coord_bytes / 1024.0,
           (double)client_comm_bytes / 1024.0,
           (double)client_total_bytes / 1024.0,
           (double)client_stack_peak_bytes / 1024.0,
           (double)server_model_bytes / 1024.0,
           (double)server_coord_bytes / 1024.0,
           (double)server_comm_bytes / 1024.0,
           (double)server_total_bytes / 1024.0,
           (double)server_stack_peak_bytes / 1024.0,
           train_ms);

    return 0;
}
