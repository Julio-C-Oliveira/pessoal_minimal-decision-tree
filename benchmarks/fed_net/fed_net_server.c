/**
 * @file fed_net_server.c
 * @brief Standalone C11 POSIX TCP Server for Federated kansho (Delivery 11.10).
 *
 * Exchanges raw HFED binary frames over real TCP sockets with K independent
 * fed_net_client OS processes. Zero heap allocation in federated core/server.
 */

#include "fed_net_common.h"
#include "ht_fed_forest.h"
#include "ht_fed_vfdt.h"

typedef enum {
    NET_MODE_FOREST = 0,
    NET_MODE_VFDT_COORD,
    NET_MODE_VFDT_MERGE
} net_mode_t;

typedef struct {
    int      fd;
    uint16_t client_id;
    uint32_t client_topo_ver;
} round_conn_t;

static ht_fed_forest_server_t g_ff_server;
static ht_fed_vfdt_server_t   g_fv_server;
static uint8_t                g_wire[FED_NET_WIRE_CAP];

static net_mode_t parse_mode(const char *s) {
    if (strcmp(s, "forest") == 0) return NET_MODE_FOREST;
    if (strcmp(s, "vfdt_merge") == 0) return NET_MODE_VFDT_MERGE;
    return NET_MODE_VFDT_COORD;
}

int main(int argc, char **argv) {
    const char *mode_str = "vfdt_coord";
    const char *task_str = "classif";
    uint16_t port = 0U;
    int n_clients = 4;
    int n_rounds = 8;
    size_t frag_bytes = 0U;
    int skip_start = 0;
    int skip_end = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode_str = argv[++i];
        else if (strcmp(argv[i], "--task") == 0 && i + 1 < argc) task_str = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-clients") == 0 && i + 1 < argc) n_clients = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) n_rounds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--fragment") == 0 && i + 1 < argc) frag_bytes = (size_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--straggler-skip") == 0 && i + 1 < argc) {
            if (sscanf(argv[++i], "%d:%d", &skip_start, &skip_end) != 2) {
                skip_start = 0;
                skip_end = 0;
            }
        }
    }

    if (n_clients < 1 || n_clients > (int)FED_NET_MAX_CLIENTS || n_rounds < 1) {
        fprintf(stderr, "Invalid server arguments\n");
        return 1;
    }

    net_mode_t mode = parse_mode(mode_str);
    bool is_reg = (strcmp(task_str, "reg") == 0);

    ht_config_params_t params;
    if (is_reg) {
        ht_default_reg_params(&params, 3U);
        params.grace_period = 40U;
        params.delta = (ht_float_t)0.01;
        params.tau = (ht_float_t)0.05;
        params.leaf_prediction = HT_LEAF_ADAPTIVE;
        params.leaf_sgd_normalize = true;
    } else {
        ht_default_params(&params, 3U, 4U);
        params.grace_period = 40U;
        params.delta = (ht_float_t)0.01;
        params.tau = (ht_float_t)0.05;
        params.criterion = HT_CRIT_GINI;
        params.leaf_prediction = HT_LEAF_NBA;
    }

    if (mode == NET_MODE_FOREST) {
        ht_fed_forest_server_cfg_t scfg;
        ht_fed_forest_default_server_config(&scfg, params.task_type);
        scfg.agg_policy = HT_FED_FOREST_AGG_ALL;
        scfg.max_forest_size = (uint8_t)HT_FED_FOREST_MAX_TREES;
        if (ht_fed_forest_server_init(&g_ff_server, &scfg) != HT_OK) {
            fprintf(stderr, "ht_fed_forest_server_init failed\n");
            return 1;
        }
    } else {
        ht_fed_vfdt_mode_t vmode = (mode == NET_MODE_VFDT_MERGE)
            ? HT_FED_VFDT_STATS_MERGE : HT_FED_VFDT_COORDINATED;
        ht_fed_vfdt_server_cfg_t scfg;
        ht_fed_vfdt_default_server_config(&scfg, vmode, &params);
        scfg.consensus = HT_FED_CONSENSUS_MAJORITY;
        scfg.min_quorum = 2U;
        scfg.threshold_agg = HT_FED_THRESH_MAX_MERIT;
        scfg.require_locally_ready = true;
        if (ht_fed_vfdt_server_init(&g_fv_server, &scfg) != HT_OK) {
            fprintf(stderr, "ht_fed_vfdt_server_init failed\n");
            return 1;
        }
    }

    /* Bind & listen on 127.0.0.1:<port> */
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }
    int opt = 1;
    (void)setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        close(listen_fd);
        return 1;
    }
    socklen_t alen = sizeof(addr);
    if (getsockname(listen_fd, (struct sockaddr *)&addr, &alen) != 0) {
        perror("getsockname");
        close(listen_fd);
        return 1;
    }
    uint16_t actual_port = ntohs(addr.sin_port);

    if (listen(listen_fd, 32) != 0) {
        perror("listen");
        close(listen_fd);
        return 1;
    }

    /* Signal orchestrator that socket is listening */
    printf("READY port=%u\n", (unsigned)actual_port);
    fflush(stdout);

    size_t total_uplink = 0U;
    size_t total_downlink = 0U;
    uint32_t crc_rejected = 0U;
    uint32_t catchups_sent = 0U;
    uint32_t snapshots_sent = 0U;
    uint32_t frames_ok = 0U;

    for (int r = 1; r <= n_rounds; ++r) {
        int expected = n_clients;
        if (skip_start > 0 && r >= skip_start && r <= skip_end && expected > 1) {
            expected = n_clients - 1;
        }

        if (mode != NET_MODE_FOREST) {
            ht_fed_vfdt_server_begin_round(&g_fv_server);
        }

        round_conn_t conns[FED_NET_MAX_CLIENTS];
        int received = 0;

        while (received < expected) {
            int cfd = accept(listen_fd, NULL, NULL);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                perror("accept");
                close(listen_fd);
                return 1;
            }

            /* Read frames until one valid HFED frame is accepted (corrupt frames rejected by CRC32) */
            bool accepted_client = false;
            for (int tries = 0; tries < 4; ++tries) {
                size_t flen = 0U;
                ht_fed_header_t hdr;
                if (!fed_tcp_recv_hfed_frame(cfd, g_wire, sizeof(g_wire), &flen, &hdr)) {
                    break;
                }
                total_uplink += flen;

                ht_status_t st = (mode == NET_MODE_FOREST)
                    ? ht_fed_forest_server_ingest(&g_ff_server, g_wire, flen)
                    : ht_fed_vfdt_server_ingest(&g_fv_server, g_wire, flen);

                if (st == HT_ERR_CORRUPT_IMAGE) {
                    crc_rejected++;
                    continue; /* Wait for client's clean retransmission on same socket */
                }
                if (st != HT_OK && st != HT_ERR_STALE_VERSION) {
                    fprintf(stderr, "Server ingest error %d from client %u in round %d\n",
                            (int)st, (unsigned)hdr.client_id, r);
                    break;
                }

                /* Both HT_OK and HT_ERR_STALE_VERSION keep the socket so stragglers receive catch-up/snapshot */
                frames_ok++;
                conns[received].fd = cfd;
                conns[received].client_id = hdr.client_id;
                conns[received].client_topo_ver = hdr.topology_version;
                received++;
                accepted_client = true;
                break;
            }

            if (!accepted_client) {
                close(cfd);
            }
        }

        /* End round aggregation & broadcast personalized HFED downlink to all round peers */
        if (mode != NET_MODE_FOREST) {
            ht_fed_vfdt_server_end_round(&g_fv_server, NULL);
        }

        for (int i = 0; i < received; ++i) {
            size_t out_len = 0U;
            ht_status_t st;
            if (mode == NET_MODE_FOREST) {
                st = ht_fed_forest_server_build_forest(&g_ff_server, conns[i].client_id, true,
                                                       g_wire, sizeof(g_wire), &out_len);
            } else {
                st = ht_fed_vfdt_server_export_update(&g_fv_server, conns[i].client_id,
                                                      conns[i].client_topo_ver,
                                                      g_wire, sizeof(g_wire), &out_len);
            }

            if (st == HT_OK && out_len >= sizeof(ht_fed_header_t)) {
                ht_fed_header_t out_hdr;
                memcpy(&out_hdr, g_wire, sizeof(out_hdr));
                if (mode != NET_MODE_FOREST) {
                    if (out_hdr.msg_type == HT_FED_MSG_VFDT_FULL_SNAPSHOT) {
                        snapshots_sent++;
                    } else if (conns[i].client_topo_ver < g_fv_server.topology_version) {
                        catchups_sent++;
                    }
                }
                if (fed_tcp_send_all(conns[i].fd, g_wire, out_len, frag_bytes)) {
                    total_downlink += out_len;
                }
            }
            close(conns[i].fd);
        }

        if (mode == NET_MODE_FOREST) {
            ht_fed_forest_server_advance_round(&g_ff_server);
        }
    }

    close(listen_fd);

    uint32_t topo_ver = (mode == NET_MODE_FOREST) ? g_ff_server.round_id : g_fv_server.topology_version;
    uint32_t g_splits = (mode == NET_MODE_FOREST) ? 0U : g_fv_server.global_tree.n_decision_nodes;
    uint32_t g_nodes = (mode == NET_MODE_FOREST) ? (uint32_t)g_ff_server.n_entries : ht_node_count(&g_fv_server.global_tree.memory);

    printf("{"
           "\"role\":\"server\","
           "\"mode\":\"%s\","
           "\"task\":\"%s\","
           "\"rounds\":%d,"
           "\"frames_ok\":%u,"
           "\"crc_rejected\":%u,"
           "\"catchups_sent\":%u,"
           "\"snapshots_sent\":%u,"
           "\"topology_version\":%u,"
           "\"global_splits\":%u,"
           "\"nodes\":%u,"
           "\"uplink_bytes\":%zu,"
           "\"downlink_bytes\":%zu"
           "}\n",
           mode_str, task_str, n_rounds,
           frames_ok, crc_rejected, catchups_sent, snapshots_sent,
           topo_ver, g_splits, g_nodes,
           total_uplink, total_downlink);
    fflush(stdout);
    return 0;
}
