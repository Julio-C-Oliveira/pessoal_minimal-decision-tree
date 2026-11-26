/**
 * @file fed_net_client.c
 * @brief Standalone C11 POSIX TCP Client process for Federated kansho (Delivery 11.10).
 *
 * Each instance executes in its own isolated OS process with static zero-malloc
 * model pools, exchanging raw HFED frames over TCP with fed_net_server.
 */

#include "fed_net_common.h"
#include "ht_fed_forest.h"
#include "ht_fed_vfdt.h"

typedef enum {
    NET_MODE_FOREST = 0,
    NET_MODE_VFDT_COORD,
    NET_MODE_VFDT_MERGE
} net_mode_t;

static ht_fed_forest_client_t g_ff_client;
static ht_fed_vfdt_client_t   g_fv_client;
static uint8_t                g_wire[FED_NET_WIRE_CAP];
static uint8_t                g_corrupt[FED_NET_WIRE_CAP];

static net_mode_t parse_mode(const char *s) {
    if (strcmp(s, "forest") == 0) return NET_MODE_FOREST;
    if (strcmp(s, "vfdt_merge") == 0) return NET_MODE_VFDT_MERGE;
    return NET_MODE_VFDT_COORD;
}

int main(int argc, char **argv) {
    const char *mode_str = "vfdt_coord";
    const char *task_str = "classif";
    const char *host = "127.0.0.1";
    uint16_t port = 0U;
    uint16_t client_id = 1U;
    int n_rounds = 8;
    int samples_per_round = 120;
    size_t frag_bytes = 0U;
    bool pure_non_iid = false;
    int inject_corrupt_round = 0;
    int skip_start = 0;
    int skip_end = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode_str = argv[++i];
        else if (strcmp(argv[i], "--task") == 0 && i + 1 < argc) task_str = argv[++i];
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--client-id") == 0 && i + 1 < argc) client_id = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) n_rounds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--samples-per-round") == 0 && i + 1 < argc) samples_per_round = atoi(argv[++i]);
        else if (strcmp(argv[i], "--fragment") == 0 && i + 1 < argc) frag_bytes = (size_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--pure-non-iid") == 0) pure_non_iid = true;
        else if (strcmp(argv[i], "--inject-corrupt-round") == 0 && i + 1 < argc) inject_corrupt_round = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip-rounds") == 0 && i + 1 < argc) {
            if (sscanf(argv[++i], "%d:%d", &skip_start, &skip_end) != 2) {
                skip_start = 0;
                skip_end = 0;
            }
        }
    }

    if (port == 0U || client_id == 0U || n_rounds < 1) {
        fprintf(stderr, "Invalid client arguments\n");
        return 1;
    }

    net_mode_t mode = parse_mode(mode_str);
    bool is_reg = (strcmp(task_str, "reg") == 0);

    ht_config_params_t params;
    if (is_reg) {
        ht_default_reg_params(&params, 3U);
        params.grace_period = 20U;
        params.delta = (ht_float_t)0.01;
        params.tau = (ht_float_t)0.05;
        params.leaf_prediction = HT_LEAF_ADAPTIVE;
        params.leaf_sgd_normalize = true;
    } else {
        ht_default_params(&params, 3U, 4U);
        params.grace_period = 20U;
        params.delta = (ht_float_t)0.01;
        params.tau = (ht_float_t)0.05;
        params.criterion = HT_CRIT_GINI;
        params.leaf_prediction = HT_LEAF_NBA;
    }

    if (mode == NET_MODE_FOREST) {
        ht_fed_forest_client_cfg_t ccfg;
        ht_fed_forest_default_client_config(&ccfg, client_id, &params);
        ccfg.voting = HT_VOTE_UNIFORM;
        if (ht_fed_forest_client_init(&g_ff_client, &ccfg) != HT_OK) {
            fprintf(stderr, "ht_fed_forest_client_init failed\n");
            return 1;
        }
    } else {
        ht_fed_vfdt_mode_t vmode = (mode == NET_MODE_VFDT_MERGE)
            ? HT_FED_VFDT_STATS_MERGE : HT_FED_VFDT_COORDINATED;
        ht_fed_vfdt_client_cfg_t ccfg;
        ht_fed_vfdt_default_client_config(&ccfg, client_id, vmode, &params);
        if (ht_fed_vfdt_client_init(&g_fv_client, &ccfg) != HT_OK) {
            fprintf(stderr, "ht_fed_vfdt_client_init failed\n");
            return 1;
        }
    }

    fed_rng_t train_rng;
    train_rng.state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)client_id * 0xBF58476D1CE4E5B9ULL);

    size_t total_uplink = 0U;
    size_t total_downlink = 0U;
    uint32_t resync_applied = 0U;
    uint32_t snapshots_applied = 0U;
    uint32_t rounds_participated = 0U;

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int r = 1; r <= n_rounds; ++r) {
        /* Local streaming training phase */
        for (int s = 0; s < samples_per_round; ++s) {
            if (is_reg) {
                ht_float_t yr = (ht_float_t)0.0;
                fed_gen_reg_sample(&train_rng, client_id, false, x, &yr);
                if (mode == NET_MODE_FOREST) {
                    ht_fed_forest_client_learn_one_reg(&g_ff_client, x, yr, (ht_float_t)1.0);
                } else {
                    ht_fed_vfdt_client_learn_one_reg(&g_fv_client, x, yr, (ht_float_t)1.0);
                }
            } else {
                uint8_t yc = 0U;
                fed_gen_quadrant_sample(&train_rng, client_id, pure_non_iid, false, x, &yc);
                if (mode == NET_MODE_FOREST) {
                    ht_fed_forest_client_learn_one(&g_ff_client, x, yc, (ht_float_t)1.0);
                } else {
                    ht_fed_vfdt_client_learn_one(&g_fv_client, x, yc, (ht_float_t)1.0);
                }
            }
        }

        /* Simulate real offline straggler during skipped rounds */
        if (skip_start > 0 && r >= skip_start && r <= skip_end) {
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = 40000000L; /* 40 ms offline window per skipped round */
            nanosleep(&ts, NULL);
            continue;
        }

        /* Export raw HFED uplink frame */
        size_t out_len = 0U;
        ht_status_t st = (mode == NET_MODE_FOREST)
            ? ht_fed_forest_client_export_update(&g_ff_client, g_wire, sizeof(g_wire), &out_len)
            : ht_fed_vfdt_client_export_update(&g_fv_client, g_wire, sizeof(g_wire), &out_len);
        if (st != HT_OK) {
            fprintf(stderr, "Client %u export failed (err=%d)\n", (unsigned)client_id, (int)st);
            return 1;
        }

        int fd = fed_tcp_connect_retry(host, port, 120);
        if (fd < 0) {
            fprintf(stderr, "Client %u failed to connect to %s:%u in round %d\n",
                    (unsigned)client_id, host, (unsigned)port, r);
            return 1;
        }

        /* Optional deliberate bit-flip corruption injection to test CRC32 over wire */
        if (inject_corrupt_round == r && out_len > sizeof(ht_fed_header_t)) {
            memcpy(g_corrupt, g_wire, out_len);
            g_corrupt[sizeof(ht_fed_header_t)] ^= 0x5AU; /* Flip payload bits */
            if (!fed_tcp_send_all(fd, g_corrupt, out_len, frag_bytes)) {
                fprintf(stderr, "Client %u failed sending corrupt test frame\n", (unsigned)client_id);
                close(fd);
                return 1;
            }
        }

        /* Send intact HFED frame (optionally fragmented across tiny TCP writes) */
        if (!fed_tcp_send_all(fd, g_wire, out_len, frag_bytes)) {
            fprintf(stderr, "Client %u failed sending uplink frame\n", (unsigned)client_id);
            close(fd);
            return 1;
        }
        total_uplink += out_len;

        /* Receive server's personalized HFED downlink frame over TCP */
        size_t in_len = 0U;
        ht_fed_header_t in_hdr;
        if (!fed_tcp_recv_hfed_frame(fd, g_wire, sizeof(g_wire), &in_len, &in_hdr)) {
            fprintf(stderr, "Client %u failed receiving downlink frame in round %d\n",
                    (unsigned)client_id, r);
            close(fd);
            return 1;
        }
        close(fd);
        total_downlink += in_len;
        rounds_participated++;

        /* Apply received HFED downlink update */
        if (mode == NET_MODE_FOREST) {
            st = ht_fed_forest_client_import_forest(&g_ff_client, g_wire, in_len);
            if (st != HT_OK) {
                fprintf(stderr, "Client %u import_forest failed (err=%d)\n", (unsigned)client_id, (int)st);
                return 1;
            }
        } else {
            uint32_t ver_before = g_fv_client.topology_version;
            st = ht_fed_vfdt_client_apply_update(&g_fv_client, g_wire, in_len);
            if (st != HT_OK) {
                fprintf(stderr, "Client %u apply_update failed (err=%d)\n", (unsigned)client_id, (int)st);
                return 1;
            }
            if (in_hdr.msg_type == HT_FED_MSG_VFDT_FULL_SNAPSHOT) {
                snapshots_applied++;
            }
            if (g_fv_client.topology_version > ver_before) {
                resync_applied++;
            }
        }
    }

    /* Evaluate final model on shared deterministic global holdout set (400 samples) */
    fed_rng_t eval_rng;
    eval_rng.state = 0x123456789ABCDEF0ULL;
    const int n_eval = 400;
    int correct = 0;
    double se_sum = 0.0;

    for (int i = 0; i < n_eval; ++i) {
        if (is_reg) {
            ht_float_t yr = (ht_float_t)0.0;
            fed_gen_reg_sample(&eval_rng, 0U, true, x, &yr);
            ht_float_t yp = (mode == NET_MODE_FOREST)
                ? ht_fed_forest_client_predict_reg(&g_ff_client, x)
                : ht_fed_vfdt_client_predict_reg(&g_fv_client, x);
            double err = (double)yp - (double)yr;
            se_sum += err * err;
        } else {
            uint8_t yc = 0U;
            fed_gen_quadrant_sample(&eval_rng, 0U, false, true, x, &yc);
            uint8_t yp = (mode == NET_MODE_FOREST)
                ? ht_fed_forest_client_predict(&g_ff_client, x)
                : ht_fed_vfdt_client_predict(&g_fv_client, x);
            if (yp == yc) {
                correct++;
            }
        }
    }

    double holdout_acc = is_reg ? 0.0 : (100.0 * (double)correct / (double)n_eval);
    double holdout_rmse = is_reg ? sqrt(se_sum / (double)n_eval) : 0.0;
    uint32_t topo_ver = (mode == NET_MODE_FOREST) ? g_ff_client.n_peers : g_fv_client.topology_version;
    uint32_t splits = (mode == NET_MODE_FOREST)
        ? g_ff_client.local_tree.n_decision_nodes
        : g_fv_client.tree.n_decision_nodes;
    uint32_t nodes = (mode == NET_MODE_FOREST)
        ? ht_node_count(&g_ff_client.local_tree.memory)
        : ht_node_count(&g_fv_client.tree.memory);

    printf("{"
           "\"role\":\"client\","
           "\"client_id\":%u,"
           "\"mode\":\"%s\","
           "\"task\":\"%s\","
           "\"rounds_participated\":%u,"
           "\"holdout_acc\":%.2f,"
           "\"holdout_rmse\":%.4f,"
           "\"topology_version\":%u,"
           "\"nodes\":%u,"
           "\"splits\":%u,"
           "\"resync_applied\":%u,"
           "\"snapshots_applied\":%u,"
           "\"uplink_bytes\":%zu,"
           "\"downlink_bytes\":%zu"
           "}\n",
           (unsigned)client_id, mode_str, task_str,
           rounds_participated, holdout_acc, holdout_rmse,
           topo_ver, nodes, splits,
           resync_applied, snapshots_applied,
           total_uplink, total_downlink);
    fflush(stdout);
    return 0;
}
