/**
 * @file ht_fed_vfdt.c
 * @brief Implementation of Mode 2 Fed-VFDT (Coordinated Splitting & Exact Stats-Merge).
 */

#include "ht_fed_vfdt.h"
#include <string.h>

typedef struct {
    uint16_t n_proposals;
    uint16_t drift_alert_node;
} ht_fed_proposal_hdr_t;

typedef struct {
    uint16_t n_leaves;
    uint16_t drift_alert_node;
    uint16_t leaf_stride;
    uint16_t reserved;
} ht_fed_leaf_stats_hdr_t;

typedef struct {
    uint16_t leaf_idx;
    uint16_t reserved;
    union {
        ht_float_t    class_counts[HT_MAX_CLASSES];
        ht_var_stat_t reg_stats;
    };
} ht_fed_leaf_stats_item_t;

typedef struct {
    uint32_t from_version;
    uint32_t to_version;
    uint16_t n_cmds;
    uint16_t reserved;
} ht_fed_broadcast_hdr_t;

/* ========================================================================= */
/* --- Internal Helpers ---------------------------------------------------- */
/* ========================================================================= */

static uint16_t ht_fed_compute_node_depth(const ht_tree_t *tree, uint16_t target_idx) {
    if (tree == NULL || tree->root_idx == HT_INVALID_IDX || target_idx >= HT_MAX_NODES) {
        return 0U;
    }
    if (target_idx == tree->root_idx) {
        return 0U;
    }

    /* Iterative DFS with depth stack (zero recursion) */
    uint16_t stack_node[128];
    uint16_t stack_depth[128];
    uint16_t top = 0U;

    stack_node[top] = tree->root_idx;
    stack_depth[top] = 0U;
    top++;

    while (top > 0U) {
        top--;
        uint16_t curr = stack_node[top];
        uint16_t d = stack_depth[top];
        if (curr == target_idx) {
            return d;
        }
        if (curr >= HT_MAX_NODES) {
            continue;
        }
        const ht_node_t *n = &tree->memory.node_pool[curr];
        if ((n->flags & HT_NODE_IN_USE) && !(n->flags & HT_NODE_IS_LEAF)) {
            if (n->right_child != HT_INVALID_IDX && top < 128U) {
                stack_node[top] = n->right_child;
                stack_depth[top] = (uint16_t)(d + 1U);
                top++;
            }
            if (n->left_child != HT_INVALID_IDX && top < 128U) {
                stack_node[top] = n->left_child;
                stack_depth[top] = (uint16_t)(d + 1U);
                top++;
            }
        }
    }
    return 0U;
}

static ht_float_t ht_fed_leaf_weight(const ht_tree_t *tree, const ht_node_t *node) {
    if (tree->params.task_type == HT_TASK_REGRESSION) {
        return node->reg_stats.count;
    }
    ht_float_t w = HT_CONST(0.0);
    for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
        w = HT_ADD(w, node->class_counts[c]);
    }
    return w;
}

static void ht_fed_server_record_cmd(ht_fed_vfdt_server_t *server, ht_fed_split_cmd_t cmd) {
    server->topology_version++;
    cmd.version = server->topology_version;
    uint32_t slot = (cmd.version - 1U) % HT_FED_SPLIT_LOG_CAP;
    server->split_log[slot] = cmd;
    if (server->topology_version > HT_FED_SPLIT_LOG_CAP) {
        server->split_log_head_version = server->topology_version - HT_FED_SPLIT_LOG_CAP + 1U;
    } else if (server->split_log_head_version == 0U) {
        server->split_log_head_version = 1U;
    }
}

static void ht_fed_server_record_drift_vote(ht_fed_vfdt_server_t *server, uint16_t node_idx) {
    if (node_idx == HT_INVALID_IDX || node_idx >= HT_MAX_NODES) {
        return;
    }
    for (uint16_t i = 0; i < server->n_drift_entries; ++i) {
        if (server->drift_vote_nodes[i] == node_idx) {
            server->drift_vote_counts[i]++;
            return;
        }
    }
    if (server->n_drift_entries < HT_FED_MAX_CLIENTS) {
        server->drift_vote_nodes[server->n_drift_entries] = node_idx;
        server->drift_vote_counts[server->n_drift_entries] = 1U;
        server->n_drift_entries++;
    }
}

static void ht_fed_merge_leaf_stats_block(ht_leaf_stats_t *dst,
                                          const ht_leaf_stats_t *src,
                                          const ht_config_params_t *params,
                                          ht_float_t dst_sgd_w,
                                          ht_float_t src_sgd_w) {
    if (params->task_type == HT_TASK_REGRESSION) {
        for (uint8_t f = 0; f < params->n_features; ++f) {
            if (params->feature_types[f] == HT_FEAT_NUMERIC) {
                ht_reg_numeric_merge(&dst->observers[f].reg_num, &src->observers[f].reg_num);
            } else {
                ht_reg_categorical_merge(&dst->observers[f].reg_cat, &src->observers[f].reg_cat);
            }
        }
        /* Sample-weighted FedAvg of linear SGD weights & intercept */
#ifdef HT_USE_FIXED_POINT
        int64_t total_w = (int64_t)dst_sgd_w + (int64_t)src_sgd_w;
        if (total_w > 0 && src_sgd_w > HT_CONST(0.0)) {
            int64_t a_q16 = ((int64_t)dst_sgd_w << 16) / total_w;
            int64_t b_q16 = ((int64_t)src_sgd_w << 16) / total_w;
            dst->reg_intercept = ht_fixed_saturate64(
                (a_q16 * (int64_t)dst->reg_intercept + b_q16 * (int64_t)src->reg_intercept) >> 16);
            dst->fmse_mean = ht_fixed_saturate64(
                (a_q16 * (int64_t)dst->fmse_mean + b_q16 * (int64_t)src->fmse_mean) >> 16);
            dst->fmse_model = ht_fixed_saturate64(
                (a_q16 * (int64_t)dst->fmse_model + b_q16 * (int64_t)src->fmse_model) >> 16);
            ht_float_t *dw = ht_leaf_reg_weights(dst, params->n_features);
            const ht_float_t *sw = ht_leaf_reg_weights_const(src, params->n_features);
            for (uint8_t f = 0; f < params->n_features; ++f) {
                dw[f] = ht_fixed_saturate64(
                    (a_q16 * (int64_t)dw[f] + b_q16 * (int64_t)sw[f]) >> 16);
            }
        }
#else
        ht_float_t total_w = HT_ADD(dst_sgd_w, src_sgd_w);
        if (total_w > HT_CONST(0.0) && src_sgd_w > HT_CONST(0.0)) {
            ht_float_t a = HT_DIV(dst_sgd_w, total_w);
            ht_float_t b = HT_DIV(src_sgd_w, total_w);
            dst->reg_intercept = HT_ADD(HT_MUL(a, dst->reg_intercept), HT_MUL(b, src->reg_intercept));
            dst->fmse_mean = HT_ADD(HT_MUL(a, dst->fmse_mean), HT_MUL(b, src->fmse_mean));
            dst->fmse_model = HT_ADD(HT_MUL(a, dst->fmse_model), HT_MUL(b, src->fmse_model));
            ht_float_t *dw = ht_leaf_reg_weights(dst, params->n_features);
            const ht_float_t *sw = ht_leaf_reg_weights_const(src, params->n_features);
            for (uint8_t f = 0; f < params->n_features; ++f) {
                dw[f] = HT_ADD(HT_MUL(a, dw[f]), HT_MUL(b, sw[f]));
            }
        }
#endif
    } else {
        for (uint8_t f = 0; f < params->n_features; ++f) {
            if (params->feature_types[f] == HT_FEAT_NUMERIC) {
                if (params->numeric_obs_type == HT_OBS_HISTOGRAM) {
                    ht_hist_merge(&dst->observers[f].hist, &src->observers[f].hist, params->n_classes);
                } else {
                    for (uint8_t c = 0; c < params->n_classes; ++c) {
                        ht_gaussian_merge(&dst->observers[f].num[c], &src->observers[f].num[c]);
                    }
                }
            } else {
                ht_categorical_merge(&dst->observers[f].cat, &src->observers[f].cat, params->n_classes);
            }
        }
        dst->mc_correct_weight = HT_ADD(dst->mc_correct_weight, src->mc_correct_weight);
        dst->nb_correct_weight = HT_ADD(dst->nb_correct_weight, src->nb_correct_weight);
    }
}

/* ========================================================================= */
/* --- Client Implementation ----------------------------------------------- */
/* ========================================================================= */

void ht_fed_vfdt_default_client_config(ht_fed_vfdt_client_cfg_t *cfg,
                                       uint16_t client_id,
                                       ht_fed_vfdt_mode_t mode,
                                       const ht_config_params_t *tree_params) {
    if (cfg == NULL) {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    if (tree_params != NULL) {
        cfg->tree_params = *tree_params;
    } else {
        ht_default_params(&cfg->tree_params, 4U, 2U);
    }
    cfg->mode = mode;
    cfg->client_id = client_id;
}

ht_status_t ht_fed_vfdt_client_init(ht_fed_vfdt_client_t *client,
                                    const ht_fed_vfdt_client_cfg_t *cfg) {
    if (client == NULL || cfg == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    memset(client, 0, sizeof(*client));
    ht_status_t st = ht_tree_init(&client->tree, &cfg->tree_params);
    if (st != HT_OK) {
        return st;
    }
    ht_tree_set_defer_splits(&client->tree, true);
    client->mode = cfg->mode;
    client->client_id = cfg->client_id;
    client->topology_version = 0U;
    client->round_id = 0U;
    client->drift_alert_node = HT_INVALID_IDX;
    return HT_OK;
}

ht_status_t ht_fed_vfdt_client_predict_proba(const ht_fed_vfdt_client_t *client,
                                             const ht_value_t x[HT_MAX_FEATURES],
                                             ht_float_t proba_out[HT_MAX_CLASSES]) {
    if (client == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    return ht_tree_predict_proba(&client->tree, x, proba_out);
}

uint8_t ht_fed_vfdt_client_predict(const ht_fed_vfdt_client_t *client,
                                   const ht_value_t x[HT_MAX_FEATURES]) {
    if (client == NULL) {
        return 0U;
    }
    return ht_tree_predict(&client->tree, x);
}

ht_status_t ht_fed_vfdt_client_learn_one(ht_fed_vfdt_client_t *client,
                                         const ht_value_t x[HT_MAX_FEATURES],
                                         uint8_t y,
                                         ht_float_t weight) {
    if (client == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    return ht_tree_learn_one(&client->tree, x, y, weight);
}

ht_float_t ht_fed_vfdt_client_predict_reg(const ht_fed_vfdt_client_t *client,
                                          const ht_value_t x[HT_MAX_FEATURES]) {
    if (client == NULL) {
        return HT_CONST(0.0);
    }
    return ht_tree_predict_reg(&client->tree, x);
}

ht_status_t ht_fed_vfdt_client_learn_one_reg(ht_fed_vfdt_client_t *client,
                                             const ht_value_t x[HT_MAX_FEATURES],
                                             ht_float_t y,
                                             ht_float_t weight) {
    if (client == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    return ht_tree_learn_one_reg(&client->tree, x, y, weight);
}

void ht_fed_vfdt_client_flag_drift(ht_fed_vfdt_client_t *client, uint16_t node_idx) {
    if (client != NULL) {
        client->drift_alert_node = node_idx;
    }
}

ht_status_t ht_fed_vfdt_client_export_update(ht_fed_vfdt_client_t *client,
                                             void *out_buf,
                                             size_t buf_cap,
                                             size_t *out_written) {
    if (client == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t flags = ht_fed_make_flags(client->tree.params.task_type);

    if (client->mode == HT_FED_VFDT_COORDINATED) {
        ht_fed_split_proposal_t props[HT_FED_MAX_PROPOSALS];
        uint16_t n_props = 0U;

        for (uint16_t i = 0; i < HT_MAX_NODES && n_props < HT_FED_MAX_PROPOSALS; ++i) {
            const ht_node_t *node = &client->tree.memory.node_pool[i];
            if (!(node->flags & HT_NODE_IN_USE) ||
                !(node->flags & HT_NODE_IS_LEAF) ||
                node->stats_idx == HT_INVALID_IDX) {
                continue;
            }
            ht_split_candidate_t best, second;
            bool should_split = false;
            if (ht_tree_evaluate_leaf_split(&client->tree, i, &best, &second, &should_split) != HT_OK) {
                continue;
            }
            if (best.feature_idx == HT_NO_FEATURE) {
                continue;
            }
            ht_float_t w = ht_fed_leaf_weight(&client->tree, node);
            if (w <= HT_CONST(0.0)) {
                continue;
            }
            uint16_t depth = ht_fed_compute_node_depth(&client->tree, i);
            if (depth >= client->tree.params.max_depth) {
                continue;
            }

            memset(&props[n_props], 0, sizeof(props[n_props]));
            props[n_props].best = best;
            props[n_props].total_weight = w;
            props[n_props].second_best_merit = second.merit;
            props[n_props].leaf_idx = i;
            props[n_props].depth = depth;
            props[n_props].locally_ready = should_split ? 1U : 0U;
            n_props++;
        }

        uint32_t payload_len = (uint32_t)(sizeof(ht_fed_proposal_hdr_t) +
                                          (size_t)n_props * sizeof(ht_fed_split_proposal_t));
        size_t total = sizeof(ht_fed_header_t) + (size_t)payload_len;
        if (out_written != NULL) {
            *out_written = total;
        }
        if (out_buf == NULL) {
            return HT_OK;
        }
        if (buf_cap < total) {
            return HT_ERR_BUFFER_TOO_SMALL;
        }

        uint8_t *ptr = (uint8_t *)out_buf + sizeof(ht_fed_header_t);
        ht_fed_proposal_hdr_t phdr;
        phdr.n_proposals = n_props;
        phdr.drift_alert_node = client->drift_alert_node;
        memcpy(ptr, &phdr, sizeof(phdr));
        ptr += sizeof(phdr);
        if (n_props > 0U) {
            memcpy(ptr, props, (size_t)n_props * sizeof(ht_fed_split_proposal_t));
        }

        client->drift_alert_node = HT_INVALID_IDX;
        return ht_fed_finalize_frame(out_buf, buf_cap, HT_FED_MSG_VFDT_SPLIT_PROPOSAL,
                                     flags, client->client_id, client->round_id,
                                     client->topology_version, payload_len, out_written);
    } else {
        /* HT_FED_VFDT_STATS_MERGE: serialize all active leaves */
        uint16_t stride = client->tree.memory.leaf_stride;
        uint16_t n_leaves = 0U;
        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            const ht_node_t *node = &client->tree.memory.node_pool[i];
            if ((node->flags & HT_NODE_IN_USE) &&
                (node->flags & HT_NODE_IS_LEAF) &&
                node->stats_idx != HT_INVALID_IDX) {
                n_leaves++;
            }
        }

        size_t item_wire_size = sizeof(ht_fed_leaf_stats_item_t) + (size_t)stride;
        uint32_t payload_len = (uint32_t)(sizeof(ht_fed_leaf_stats_hdr_t) +
                                          (size_t)n_leaves * item_wire_size);
        size_t total = sizeof(ht_fed_header_t) + (size_t)payload_len;
        if (out_written != NULL) {
            *out_written = total;
        }
        if (out_buf == NULL) {
            return HT_OK;
        }
        if (buf_cap < total) {
            return HT_ERR_BUFFER_TOO_SMALL;
        }

        uint8_t *ptr = (uint8_t *)out_buf + sizeof(ht_fed_header_t);
        ht_fed_leaf_stats_hdr_t shdr;
        shdr.n_leaves = n_leaves;
        shdr.drift_alert_node = client->drift_alert_node;
        shdr.leaf_stride = stride;
        shdr.reserved = 0U;
        memcpy(ptr, &shdr, sizeof(shdr));
        ptr += sizeof(shdr);

        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            const ht_node_t *node = &client->tree.memory.node_pool[i];
            if ((node->flags & HT_NODE_IN_USE) &&
                (node->flags & HT_NODE_IS_LEAF) &&
                node->stats_idx != HT_INVALID_IDX) {
                ht_fed_leaf_stats_item_t item;
                memset(&item, 0, sizeof(item));
                item.leaf_idx = i;
                if (client->tree.params.task_type == HT_TASK_REGRESSION) {
                    item.reg_stats = node->reg_stats;
                } else {
                    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
                        item.class_counts[c] = node->class_counts[c];
                    }
                }
                memcpy(ptr, &item, sizeof(item));
                ptr += sizeof(item);

                const ht_leaf_stats_t *st = ht_stats_get_const(&client->tree.memory, node->stats_idx);
                memcpy(ptr, st, stride);
                ptr += stride;
            }
        }

        client->drift_alert_node = HT_INVALID_IDX;
        return ht_fed_finalize_frame(out_buf, buf_cap, HT_FED_MSG_VFDT_LEAF_STATS,
                                     flags, client->client_id, client->round_id,
                                     client->topology_version, payload_len, out_written);
    }
}

ht_status_t ht_fed_vfdt_client_apply_update(ht_fed_vfdt_client_t *client,
                                            const void *buf,
                                            size_t buf_len) {
    if (client == NULL || buf == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    ht_fed_header_t hdr;
    const uint8_t *payload = NULL;
    ht_status_t st = ht_fed_parse_frame(buf, buf_len, &hdr, &payload);
    if (st != HT_OK) {
        return st;
    }
    if (hdr.flags != ht_fed_make_flags(client->tree.params.task_type)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    if (hdr.msg_type == HT_FED_MSG_VFDT_FULL_SNAPSHOT) {
        st = ht_tree_load(&client->tree, payload, hdr.payload_len);
        if (st != HT_OK) {
            return st;
        }
        ht_tree_set_defer_splits(&client->tree, true);
        /* Reactivate leaves so the resynced straggler resumes collecting statistics */
        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            const ht_node_t *n = &client->tree.memory.node_pool[i];
            if ((n->flags & HT_NODE_IN_USE) &&
                (n->flags & HT_NODE_IS_LEAF) &&
                n->stats_idx == HT_INVALID_IDX) {
                (void)ht_tree_reactivate_leaf(&client->tree, i);
            }
        }
        client->topology_version = hdr.topology_version;
        client->round_id = hdr.round_id;
        return HT_OK;
    }

    if (hdr.msg_type == HT_FED_MSG_VFDT_SPLIT_BROADCAST) {
        if (hdr.payload_len < sizeof(ht_fed_broadcast_hdr_t)) {
            return HT_ERR_CORRUPT_IMAGE;
        }
        ht_fed_broadcast_hdr_t bhdr;
        memcpy(&bhdr, payload, sizeof(bhdr));
        size_t expected = sizeof(bhdr) + (size_t)bhdr.n_cmds * sizeof(ht_fed_split_cmd_t);
        if (hdr.payload_len < expected) {
            return HT_ERR_CORRUPT_IMAGE;
        }

        if (bhdr.from_version > client->topology_version) {
            return HT_ERR_STALE_VERSION;
        }

        const uint8_t *ptr = payload + sizeof(bhdr);
        for (uint16_t i = 0; i < bhdr.n_cmds; ++i) {
            ht_fed_split_cmd_t cmd;
            memcpy(&cmd, ptr, sizeof(cmd));
            ptr += sizeof(cmd);

            if (cmd.version <= client->topology_version) {
                continue; /* Already applied */
            }
            if (cmd.version != client->topology_version + 1U) {
                return HT_ERR_STALE_VERSION;
            }

            if (cmd.cmd_type == HT_FED_CMD_RESET_SUBTREE) {
                st = ht_tree_reset_subtree_to_leaf(&client->tree, cmd.node_idx);
            } else {
                st = ht_tree_apply_split(&client->tree, cmd.node_idx, &cmd.cand, cmd.depth, NULL, NULL);
            }
            if (st != HT_OK) {
                return st;
            }
            client->topology_version = cmd.version;
        }
        client->round_id = hdr.round_id;
        return HT_OK;
    }

    return HT_ERR_CORRUPT_IMAGE;
}

/* ========================================================================= */
/* --- Server Implementation ----------------------------------------------- */
/* ========================================================================= */

void ht_fed_vfdt_default_server_config(ht_fed_vfdt_server_cfg_t *cfg,
                                       ht_fed_vfdt_mode_t mode,
                                       const ht_config_params_t *tree_params) {
    if (cfg == NULL) {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    if (tree_params != NULL) {
        cfg->tree_params = *tree_params;
    } else {
        ht_default_params(&cfg->tree_params, 4U, 2U);
    }
    cfg->mode = mode;
    cfg->consensus = HT_FED_CONSENSUS_QUORUM;
    cfg->threshold_agg = HT_FED_THRESH_MAX_MERIT;
    cfg->min_quorum = 2U;
    cfg->min_drift_quorum = 2U;
    cfg->max_splits_per_round = 0U;
    cfg->require_locally_ready = true;
}

ht_status_t ht_fed_vfdt_server_init(ht_fed_vfdt_server_t *server,
                                    const ht_fed_vfdt_server_cfg_t *cfg) {
    if (server == NULL || cfg == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    memset(server, 0, sizeof(*server));
    server->cfg = *cfg;
    if (server->cfg.min_quorum == 0U) {
        server->cfg.min_quorum = 1U;
    }
    if (server->cfg.min_drift_quorum == 0U) {
        server->cfg.min_drift_quorum = 1U;
    }

    ht_status_t st = ht_tree_init(&server->global_tree, &cfg->tree_params);
    if (st != HT_OK) {
        return st;
    }
    ht_tree_set_defer_splits(&server->global_tree, true);
    server->topology_version = 0U;
    server->split_log_head_version = 1U;
    server->round_id = 0U;
    return HT_OK;
}

ht_status_t ht_fed_vfdt_server_begin_round(ht_fed_vfdt_server_t *server) {
    if (server == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    server->round_id++;
    server->n_round_proposals = 0U;
    server->n_round_clients = 0U;
    server->n_drift_entries = 0U;
    server->n_stale_rejected = 0U;

    /* Stateless round re-merge: zero active leaf scratch stats blocks on global_tree */
    if (server->cfg.mode == HT_FED_VFDT_STATS_MERGE) {
        memset(server->merge_leaf_delta, 0, sizeof(server->merge_leaf_delta));
        memset(server->merge_sgd_weight_sum, 0, sizeof(server->merge_sgd_weight_sum));

        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            const ht_node_t *n = &server->global_tree.memory.node_pool[i];
            if ((n->flags & HT_NODE_IN_USE) &&
                (n->flags & HT_NODE_IS_LEAF) &&
                n->stats_idx != HT_INVALID_IDX) {
                ht_leaf_stats_t *st = ht_stats_get(&server->global_tree.memory, n->stats_idx);
                if (st != NULL) {
                    memset(st, 0, server->global_tree.memory.leaf_stride);
                }
            }
        }
    }

    return HT_OK;
}

ht_status_t ht_fed_vfdt_server_ingest(ht_fed_vfdt_server_t *server,
                                      const void *buf,
                                      size_t buf_len) {
    if (server == NULL || buf == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    ht_fed_header_t hdr;
    const uint8_t *payload = NULL;
    ht_status_t st = ht_fed_parse_frame(buf, buf_len, &hdr, &payload);
    if (st != HT_OK) {
        return st;
    }
    if (hdr.flags != ht_fed_make_flags(server->cfg.tree_params.task_type)) {
        return HT_ERR_CORRUPT_IMAGE;
    }
    if (hdr.topology_version != server->topology_version) {
        server->n_stale_rejected++;
        return HT_ERR_STALE_VERSION;
    }

    if (server->cfg.mode == HT_FED_VFDT_COORDINATED) {
        if (hdr.msg_type != HT_FED_MSG_VFDT_SPLIT_PROPOSAL ||
            hdr.payload_len < sizeof(ht_fed_proposal_hdr_t)) {
            return HT_ERR_CORRUPT_IMAGE;
        }
        ht_fed_proposal_hdr_t phdr;
        memcpy(&phdr, payload, sizeof(phdr));
        size_t expected = sizeof(phdr) + (size_t)phdr.n_proposals * sizeof(ht_fed_split_proposal_t);
        if (hdr.payload_len < expected) {
            return HT_ERR_CORRUPT_IMAGE;
        }

        ht_fed_server_record_drift_vote(server, phdr.drift_alert_node);

        const uint8_t *ptr = payload + sizeof(phdr);
        for (uint16_t i = 0; i < phdr.n_proposals; ++i) {
            if (server->n_round_proposals < HT_FED_SERVER_MAX_PROPOSALS) {
                memcpy(&server->round_proposals[server->n_round_proposals],
                       ptr, sizeof(ht_fed_split_proposal_t));
                server->n_round_proposals++;
            }
            ptr += sizeof(ht_fed_split_proposal_t);
        }
        server->n_round_clients++;
        return HT_OK;
    } else {
        /* HT_FED_VFDT_STATS_MERGE */
        if (hdr.msg_type != HT_FED_MSG_VFDT_LEAF_STATS ||
            hdr.payload_len < sizeof(ht_fed_leaf_stats_hdr_t)) {
            return HT_ERR_CORRUPT_IMAGE;
        }
        ht_fed_leaf_stats_hdr_t shdr;
        memcpy(&shdr, payload, sizeof(shdr));
        if (shdr.leaf_stride != server->global_tree.memory.leaf_stride) {
            return HT_ERR_CORRUPT_IMAGE;
        }
        size_t item_wire = sizeof(ht_fed_leaf_stats_item_t) + (size_t)shdr.leaf_stride;
        size_t expected = sizeof(shdr) + (size_t)shdr.n_leaves * item_wire;
        if (hdr.payload_len < expected) {
            return HT_ERR_CORRUPT_IMAGE;
        }

        ht_fed_server_record_drift_vote(server, shdr.drift_alert_node);

        const uint8_t *ptr = payload + sizeof(shdr);
        bool is_reg = (server->cfg.tree_params.task_type == HT_TASK_REGRESSION);

        /* Aligned scratch buffer for one incoming leaf_stats_t */
        ht_leaf_stats_t incoming_stats;

        for (uint16_t k = 0; k < shdr.n_leaves; ++k) {
            ht_fed_leaf_stats_item_t item;
            memcpy(&item, ptr, sizeof(item));
            ptr += sizeof(item);

            memset(&incoming_stats, 0, sizeof(incoming_stats));
            memcpy(&incoming_stats, ptr, shdr.leaf_stride);
            ptr += shdr.leaf_stride;

            if (item.leaf_idx >= HT_MAX_NODES) {
                continue;
            }
            const ht_node_t *gnode = &server->global_tree.memory.node_pool[item.leaf_idx];
            if (!(gnode->flags & HT_NODE_IN_USE) ||
                !(gnode->flags & HT_NODE_IS_LEAF) ||
                gnode->stats_idx == HT_INVALID_IDX) {
                continue;
            }

            uint16_t s_idx = gnode->stats_idx;
            ht_leaf_stats_t *dst_stats = ht_stats_get(&server->global_tree.memory, s_idx);
            if (dst_stats == NULL) {
                continue;
            }

            ht_float_t client_local_w = HT_CONST(0.0);
            if (is_reg) {
                ht_var_stat_t delta_reg = ht_var_subtract(&item.reg_stats, &gnode->reg_stats);
                client_local_w = delta_reg.count;
                server->merge_leaf_delta[s_idx].reg_stats =
                    ht_var_combine(&server->merge_leaf_delta[s_idx].reg_stats, &delta_reg);
            } else {
                for (uint8_t c = 0; c < server->cfg.tree_params.n_classes; ++c) {
                    ht_float_t d = HT_SUB(item.class_counts[c], gnode->class_counts[c]);
                    if (d > HT_CONST(0.0)) {
                        server->merge_leaf_delta[s_idx].class_counts[c] =
                            HT_ADD(server->merge_leaf_delta[s_idx].class_counts[c], d);
                        client_local_w = HT_ADD(client_local_w, d);
                    }
                }
            }

            ht_fed_merge_leaf_stats_block(dst_stats,
                                          &incoming_stats,
                                          &server->cfg.tree_params,
                                          server->merge_sgd_weight_sum[s_idx],
                                          client_local_w);
            server->merge_sgd_weight_sum[s_idx] = HT_ADD(server->merge_sgd_weight_sum[s_idx], client_local_w);
        }

        server->n_round_clients++;
        return HT_OK;
    }
}

typedef struct {
    uint16_t             leaf_idx;
    uint16_t             depth;
    uint16_t             winning_votes;
    ht_float_t           winning_max_merit;
    ht_split_candidate_t cand;
} ht_fed_pending_leaf_split_t;

static ht_status_t ht_fed_vfdt_end_round_coordinated(ht_fed_vfdt_server_t *server,
                                                     uint16_t *cmds_applied) {
    /* Identify distinct proposed leaf_idx values in deterministic ascending order */
    uint16_t unique_leaves[HT_FED_SERVER_MAX_PROPOSALS];
    uint16_t n_unique = 0U;

    for (uint16_t i = 0; i < server->n_round_proposals; ++i) {
        uint16_t l_idx = server->round_proposals[i].leaf_idx;
        bool seen = false;
        for (uint16_t j = 0; j < n_unique; ++j) {
            if (unique_leaves[j] == l_idx) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            unique_leaves[n_unique++] = l_idx;
        }
    }

    /* Sort unique_leaves ascending for deterministic topology evolution */
    for (uint16_t i = 1U; i < n_unique; ++i) {
        uint16_t key = unique_leaves[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && unique_leaves[j] > key) {
            unique_leaves[j + 1] = unique_leaves[j];
            j--;
        }
        unique_leaves[j + 1] = key;
    }

    bool is_reg = (server->cfg.tree_params.task_type == HT_TASK_REGRESSION);
    ht_fed_pending_leaf_split_t ready_splits[HT_FED_SERVER_MAX_PROPOSALS];
    uint16_t n_ready_splits = 0U;

    for (uint16_t u = 0; u < n_unique; ++u) {
        uint16_t leaf_idx = unique_leaves[u];
        if (leaf_idx >= HT_MAX_NODES) {
            continue;
        }
        const ht_node_t *gnode = &server->global_tree.memory.node_pool[leaf_idx];
        if (!(gnode->flags & HT_NODE_IN_USE) || !(gnode->flags & HT_NODE_IS_LEAF)) {
            continue;
        }

        uint16_t   feat_votes[HT_MAX_FEATURES];
        ht_float_t feat_merit_sum[HT_MAX_FEATURES];
#ifdef HT_USE_FIXED_POINT
        int64_t    feat_wmerit_sum[HT_MAX_FEATURES];
#else
        ht_float_t feat_wmerit_sum[HT_MAX_FEATURES];
#endif
        ht_float_t feat_max_merit[HT_MAX_FEATURES];
        uint16_t   feat_best_prop[HT_MAX_FEATURES];

        for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
            feat_votes[f] = 0U;
            feat_merit_sum[f] = HT_CONST(0.0);
#ifdef HT_USE_FIXED_POINT
            feat_wmerit_sum[f] = 0;
#else
            feat_wmerit_sum[f] = HT_CONST(0.0);
#endif
            feat_max_merit[f] = HT_NEG_INFINITY;
            feat_best_prop[f] = HT_INVALID_IDX;
        }

        uint16_t eligible_props = 0U;
        uint16_t depth = 0U;

        for (uint16_t i = 0; i < server->n_round_proposals; ++i) {
            const ht_fed_split_proposal_t *p = &server->round_proposals[i];
            if (p->leaf_idx != leaf_idx) {
                continue;
            }
            if (p->best.feature_idx >= server->cfg.tree_params.n_features) {
                continue;
            }
            if (server->cfg.require_locally_ready && !p->locally_ready) {
                continue;
            }

            depth = p->depth;
            eligible_props++;

            uint8_t f = p->best.feature_idx;
            ht_float_t w = (p->total_weight > HT_CONST(0.0)) ? p->total_weight : HT_CONST(1.0);
            ht_float_t m = p->best.merit;
            ht_float_t m_pos = (m > HT_CONST(0.0)) ? m : HT_CONST(0.0);

            feat_votes[f]++;
            feat_merit_sum[f] = HT_ADD(feat_merit_sum[f], m_pos);
#ifdef HT_USE_FIXED_POINT
            feat_wmerit_sum[f] += (int64_t)w * (int64_t)m_pos;
#else
            feat_wmerit_sum[f] = HT_ADD(feat_wmerit_sum[f], HT_MUL(w, m_pos));
#endif
            if (feat_best_prop[f] == HT_INVALID_IDX || m > feat_max_merit[f]) {
                feat_max_merit[f] = m;
                feat_best_prop[f] = i;
            }
        }

        if (eligible_props < server->cfg.min_quorum) {
            continue;
        }

        uint8_t winner_f = HT_NO_FEATURE;
        uint16_t best_votes = 0U;
        ht_float_t best_primary = HT_NEG_INFINITY;
#ifdef HT_USE_FIXED_POINT
        int64_t best_wmerit = INT64_MIN;
#endif
        ht_float_t best_tie = HT_NEG_INFINITY;

        for (uint8_t f = 0; f < server->cfg.tree_params.n_features; ++f) {
            if (feat_votes[f] == 0U) {
                continue;
            }

            if (server->cfg.consensus == HT_FED_CONSENSUS_QUORUM) {
                /* Paper-faithful QUORUM: require per-feature support >= min_quorum;
                 * among qualified features, pick highest merit sum (tie-break: max merit). */
                if (feat_votes[f] < server->cfg.min_quorum) {
                    continue;
                }
                if (feat_merit_sum[f] > best_primary ||
                    (feat_merit_sum[f] == best_primary && feat_max_merit[f] > best_tie)) {
                    best_primary = feat_merit_sum[f];
                    best_tie = feat_max_merit[f];
                    best_votes = feat_votes[f];
                    winner_f = f;
                }
            } else if (server->cfg.consensus == HT_FED_CONSENSUS_MERIT) {
                /* Paper-faithful BEST-MERIT: highest individual merit across eligible proposals */
                if (feat_max_merit[f] > best_primary ||
                    (feat_max_merit[f] == best_primary && feat_votes[f] > best_votes)) {
                    best_primary = feat_max_merit[f];
                    best_tie = feat_max_merit[f];
                    best_votes = feat_votes[f];
                    winner_f = f;
                }
            } else if (server->cfg.consensus == HT_FED_CONSENSUS_WEIGHTED_MERIT) {
                /* Hybrid WEIGHTED-MERIT: highest sum(w_k * G_k(f)) */
#ifdef HT_USE_FIXED_POINT
                if (feat_wmerit_sum[f] > best_wmerit ||
                    (feat_wmerit_sum[f] == best_wmerit && feat_max_merit[f] > best_tie)) {
                    best_wmerit = feat_wmerit_sum[f];
                    best_tie = feat_max_merit[f];
                    best_votes = feat_votes[f];
                    winner_f = f;
                }
#else
                if (feat_wmerit_sum[f] > best_primary ||
                    (feat_wmerit_sum[f] == best_primary && feat_max_merit[f] > best_tie)) {
                    best_primary = feat_wmerit_sum[f];
                    best_tie = feat_max_merit[f];
                    best_votes = feat_votes[f];
                    winner_f = f;
                }
#endif
            } else {
                /* Paper-faithful MAJORITY-VOTE (Plurality): most votes, tie-break by highest individual merit */
                if (feat_votes[f] > best_votes ||
                    (feat_votes[f] == best_votes && feat_max_merit[f] > best_tie)) {
                    best_votes = feat_votes[f];
                    best_tie = feat_max_merit[f];
                    best_primary = feat_max_merit[f];
                    winner_f = f;
                }
            }
        }

        if (winner_f == HT_NO_FEATURE || feat_best_prop[winner_f] == HT_INVALID_IDX) {
            continue;
        }

        /* Aggregate children stats across all eligible proposals matching winner_f */
        ht_split_candidate_t global_cand;
        ht_split_candidate_init(&global_cand);
        global_cand.feature_idx = winner_f;
        global_cand.is_categorical =
            (server->cfg.tree_params.feature_types[winner_f] == HT_FEAT_CATEGORICAL) ? 1U : 0U;

#ifdef HT_USE_FIXED_POINT
        int64_t thresh_num = 0;
        int64_t thresh_den = 0;
#else
        ht_float_t thresh_num = HT_CONST(0.0);
        ht_float_t thresh_den = HT_CONST(0.0);
#endif
        ht_float_t cat_votes[HT_MAX_CATEGORIES];
        memset(cat_votes, 0, sizeof(cat_votes));

        for (uint16_t i = 0; i < server->n_round_proposals; ++i) {
            const ht_fed_split_proposal_t *p = &server->round_proposals[i];
            if (p->leaf_idx != leaf_idx || p->best.feature_idx != winner_f) {
                continue;
            }
            if (server->cfg.require_locally_ready && !p->locally_ready) {
                continue;
            }

            ht_float_t w = (p->total_weight > HT_CONST(0.0)) ? p->total_weight : HT_CONST(1.0);
            if (global_cand.is_categorical) {
                int32_t cv = p->best.split_val.cat_val;
                if (cv >= 0 && cv < (int32_t)HT_MAX_CATEGORIES) {
                    cat_votes[cv] = HT_ADD(cat_votes[cv], w);
                }
            } else {
#ifdef HT_USE_FIXED_POINT
                thresh_num += (int64_t)w * (int64_t)p->best.split_val.threshold;
                thresh_den += (int64_t)w;
#else
                thresh_num = HT_ADD(thresh_num, HT_MUL(w, p->best.split_val.threshold));
                thresh_den = HT_ADD(thresh_den, w);
#endif
            }

            if (is_reg) {
                global_cand.lhs_var = ht_var_combine(&global_cand.lhs_var, &p->best.lhs_var);
                global_cand.rhs_var = ht_var_combine(&global_cand.rhs_var, &p->best.rhs_var);
            } else {
                for (uint8_t c = 0; c < server->cfg.tree_params.n_classes; ++c) {
                    global_cand.lhs_dist[c] = HT_ADD(global_cand.lhs_dist[c], p->best.lhs_dist[c]);
                    global_cand.rhs_dist[c] = HT_ADD(global_cand.rhs_dist[c], p->best.rhs_dist[c]);
                }
            }
        }

        const ht_fed_split_proposal_t *best_p = &server->round_proposals[feat_best_prop[winner_f]];
        if (server->cfg.threshold_agg == HT_FED_THRESH_MAX_MERIT) {
            /* Paper-faithful: exact threshold/category from highest-merit winning proposal */
            global_cand.split_val = best_p->best.split_val;
            global_cand.merit = best_p->best.merit;
        } else {
            /* Sample-weighted mean threshold / weighted mode category */
            if (global_cand.is_categorical) {
                int32_t best_cat = best_p->best.split_val.cat_val;
                ht_float_t best_cw = HT_CONST(-1.0);
                for (int32_t cv = 0; cv < (int32_t)HT_MAX_CATEGORIES; ++cv) {
                    if (cat_votes[cv] > best_cw) {
                        best_cw = cat_votes[cv];
                        best_cat = cv;
                    }
                }
                global_cand.split_val.cat_val = best_cat;
            } else {
#ifdef HT_USE_FIXED_POINT
                global_cand.split_val.threshold = (thresh_den > 0)
                    ? ht_fixed_saturate64(thresh_num / thresh_den) : best_p->best.split_val.threshold;
#else
                global_cand.split_val.threshold = (thresh_den > HT_CONST(0.0))
                    ? HT_DIV(thresh_num, thresh_den) : best_p->best.split_val.threshold;
#endif
            }
#ifdef HT_USE_FIXED_POINT
            global_cand.merit = (thresh_den > 0)
                ? ht_fixed_saturate64(feat_wmerit_sum[winner_f] / thresh_den) : best_p->best.merit;
#else
            global_cand.merit = (thresh_den > HT_CONST(0.0))
                ? HT_DIV(feat_wmerit_sum[winner_f], thresh_den) : best_p->best.merit;
#endif
        }

        ready_splits[n_ready_splits].leaf_idx = leaf_idx;
        ready_splits[n_ready_splits].depth = depth;
        ready_splits[n_ready_splits].winning_votes = best_votes;
        ready_splits[n_ready_splits].winning_max_merit = feat_max_merit[winner_f];
        ready_splits[n_ready_splits].cand = global_cand;
        n_ready_splits++;
    }

    /* When max_splits_per_round > 0 (e.g. 1 to match paper's 1-split/round),
     * prioritize leaves by highest vote count, tie-breaking by highest merit. */
    uint16_t to_apply = n_ready_splits;
    if (server->cfg.max_splits_per_round > 0U && n_ready_splits > server->cfg.max_splits_per_round) {
        for (uint16_t i = 1U; i < n_ready_splits; ++i) {
            ht_fed_pending_leaf_split_t key = ready_splits[i];
            int32_t j = (int32_t)i - 1;
            while (j >= 0) {
                bool key_better = (key.winning_votes > ready_splits[j].winning_votes) ||
                                  (key.winning_votes == ready_splits[j].winning_votes &&
                                   key.winning_max_merit > ready_splits[j].winning_max_merit);
                if (!key_better) {
                    break;
                }
                ready_splits[j + 1] = ready_splits[j];
                j--;
            }
            ready_splits[j + 1] = key;
        }
        to_apply = server->cfg.max_splits_per_round;
    }

    for (uint16_t k = 0; k < to_apply; ++k) {
        ht_status_t st = ht_tree_apply_split(&server->global_tree,
                                             ready_splits[k].leaf_idx,
                                             &ready_splits[k].cand,
                                             ready_splits[k].depth,
                                             NULL, NULL);
        if (st == HT_OK) {
            ht_fed_split_cmd_t cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.cmd_type = (uint8_t)HT_FED_CMD_SPLIT;
            cmd.node_idx = ready_splits[k].leaf_idx;
            cmd.depth = ready_splits[k].depth;
            cmd.cand = ready_splits[k].cand;
            ht_fed_server_record_cmd(server, cmd);
            (*cmds_applied)++;
        }
    }

    return HT_OK;
}

static ht_status_t ht_fed_vfdt_end_round_stats_merge(ht_fed_vfdt_server_t *server,
                                                     uint16_t *cmds_applied) {
    bool is_reg = (server->cfg.tree_params.task_type == HT_TASK_REGRESSION);

    /* Snapshot currently active leaves so newly created children are not evaluated in the same round */
    uint16_t leaves[HT_MAX_ACTIVE_LEAVES];
    uint16_t n_leaves = 0U;
    for (uint16_t i = 0; i < HT_MAX_NODES && n_leaves < HT_MAX_ACTIVE_LEAVES; ++i) {
        const ht_node_t *n = &server->global_tree.memory.node_pool[i];
        if ((n->flags & HT_NODE_IN_USE) &&
            (n->flags & HT_NODE_IS_LEAF) &&
            n->stats_idx != HT_INVALID_IDX) {
            leaves[n_leaves++] = i;
        }
    }

    for (uint16_t k = 0; k < n_leaves; ++k) {
        uint16_t leaf_idx = leaves[k];
        ht_node_t *node = &server->global_tree.memory.node_pool[leaf_idx];
        uint16_t s_idx = node->stats_idx;
        if (s_idx == HT_INVALID_IDX) {
            continue;
        }

        uint16_t depth = ht_fed_compute_node_depth(&server->global_tree, leaf_idx);
        if (depth >= server->cfg.tree_params.max_depth) {
            continue;
        }

        /* Temporarily combine post-split seed with this round's merged local deltas */
        ht_float_t saved_cc[HT_MAX_CLASSES];
        ht_var_stat_t saved_reg = node->reg_stats;
        ht_float_t delta_w = HT_CONST(0.0);

        if (is_reg) {
            delta_w = server->merge_leaf_delta[s_idx].reg_stats.count;
            if (delta_w < HT_FROM_UINT(server->cfg.tree_params.grace_period)) {
                continue;
            }
            node->reg_stats = ht_var_combine(&saved_reg, &server->merge_leaf_delta[s_idx].reg_stats);
        } else {
            for (uint8_t c = 0; c < server->cfg.tree_params.n_classes; ++c) {
                saved_cc[c] = node->class_counts[c];
                delta_w = HT_ADD(delta_w, server->merge_leaf_delta[s_idx].class_counts[c]);
                node->class_counts[c] = HT_ADD(saved_cc[c], server->merge_leaf_delta[s_idx].class_counts[c]);
            }
            if (delta_w < HT_FROM_UINT(server->cfg.tree_params.grace_period)) {
                for (uint8_t c = 0; c < server->cfg.tree_params.n_classes; ++c) {
                    node->class_counts[c] = saved_cc[c];
                }
                continue;
            }
        }

        ht_split_candidate_t best;
        bool should_split = false;
        ht_status_t st = ht_tree_evaluate_leaf_split(&server->global_tree, leaf_idx, &best, NULL, &should_split);

        if (st == HT_OK && should_split && best.feature_idx != HT_NO_FEATURE) {
            st = ht_tree_apply_split(&server->global_tree, leaf_idx, &best, depth, NULL, NULL);
            if (st == HT_OK) {
                ht_fed_split_cmd_t cmd;
                memset(&cmd, 0, sizeof(cmd));
                cmd.cmd_type = (uint8_t)HT_FED_CMD_SPLIT;
                cmd.node_idx = leaf_idx;
                cmd.depth = depth;
                cmd.cand = best;
                ht_fed_server_record_cmd(server, cmd);
                (*cmds_applied)++;
                continue;
            }
        }

        /* Restore pure post-split seed so subsequent stateless rounds never double-count */
        if (is_reg) {
            node->reg_stats = saved_reg;
        } else {
            for (uint8_t c = 0; c < server->cfg.tree_params.n_classes; ++c) {
                node->class_counts[c] = saved_cc[c];
            }
        }
    }

    return HT_OK;
}

ht_status_t ht_fed_vfdt_server_end_round(ht_fed_vfdt_server_t *server,
                                         uint16_t *out_cmds_applied) {
    if (server == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t cmds_applied = 0U;

    /* 1. Process coordinated concept-drift resets first */
    for (uint16_t i = 0; i < server->n_drift_entries; ++i) {
        if (server->drift_vote_counts[i] >= server->cfg.min_drift_quorum) {
            uint16_t target = server->drift_vote_nodes[i];
            if (target < HT_MAX_NODES &&
                (server->global_tree.memory.node_pool[target].flags & HT_NODE_IN_USE)) {
                if (ht_tree_reset_subtree_to_leaf(&server->global_tree, target) == HT_OK) {
                    ht_fed_split_cmd_t cmd;
                    memset(&cmd, 0, sizeof(cmd));
                    cmd.cmd_type = (uint8_t)HT_FED_CMD_RESET_SUBTREE;
                    cmd.node_idx = target;
                    ht_fed_server_record_cmd(server, cmd);
                    cmds_applied++;
                }
            }
        }
    }

    /* 2. Evaluate splits according to configured mode */
    ht_status_t st = HT_OK;
    if (cmds_applied == 0U) {
        if (server->cfg.mode == HT_FED_VFDT_COORDINATED) {
            st = ht_fed_vfdt_end_round_coordinated(server, &cmds_applied);
        } else {
            st = ht_fed_vfdt_end_round_stats_merge(server, &cmds_applied);
        }
    }

    if (out_cmds_applied != NULL) {
        *out_cmds_applied = cmds_applied;
    }
    return st;
}

ht_status_t ht_fed_vfdt_server_export_update(const ht_fed_vfdt_server_t *server,
                                             uint16_t target_client_id,
                                             uint32_t client_topology_version,
                                             void *out_buf,
                                             size_t buf_cap,
                                             size_t *out_written) {
    if (server == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t flags = ht_fed_make_flags(server->cfg.tree_params.task_type);

    /* Determine whether incremental catch-up via split_log is possible */
    bool need_snapshot = false;
    uint32_t n_cmds = 0U;

    if (client_topology_version > server->topology_version) {
        need_snapshot = true;
    } else {
        n_cmds = server->topology_version - client_topology_version;
        if (n_cmds > HT_FED_SPLIT_LOG_CAP) {
            need_snapshot = true;
        } else if (n_cmds > 0U && (client_topology_version + 1U) < server->split_log_head_version) {
            need_snapshot = true;
        }
    }

    if (need_snapshot) {
        size_t snap_bytes = 0U;
        ht_status_t st = ht_tree_save(&server->global_tree, NULL, 0U, &snap_bytes, false);
        if (st != HT_OK) {
            return st;
        }
        size_t total = sizeof(ht_fed_header_t) + snap_bytes;
        if (out_written != NULL) {
            *out_written = total;
        }
        if (out_buf == NULL) {
            return HT_OK;
        }
        if (buf_cap < total) {
            return HT_ERR_BUFFER_TOO_SMALL;
        }

        size_t written_snap = 0U;
        st = ht_tree_save(&server->global_tree,
                          (uint8_t *)out_buf + sizeof(ht_fed_header_t),
                          snap_bytes,
                          &written_snap,
                          false);
        if (st != HT_OK) {
            return st;
        }

        return ht_fed_finalize_frame(out_buf, buf_cap, HT_FED_MSG_VFDT_FULL_SNAPSHOT,
                                     flags, target_client_id, server->round_id,
                                     server->topology_version, (uint32_t)written_snap, out_written);
    }

    /* Incremental SPLIT_BROADCAST */
    uint32_t payload_len = (uint32_t)(sizeof(ht_fed_broadcast_hdr_t) +
                                      (size_t)n_cmds * sizeof(ht_fed_split_cmd_t));
    size_t total = sizeof(ht_fed_header_t) + (size_t)payload_len;
    if (out_written != NULL) {
        *out_written = total;
    }
    if (out_buf == NULL) {
        return HT_OK;
    }
    if (buf_cap < total) {
        return HT_ERR_BUFFER_TOO_SMALL;
    }

    uint8_t *ptr = (uint8_t *)out_buf + sizeof(ht_fed_header_t);
    ht_fed_broadcast_hdr_t bhdr;
    bhdr.from_version = client_topology_version;
    bhdr.to_version = server->topology_version;
    bhdr.n_cmds = (uint16_t)n_cmds;
    bhdr.reserved = 0U;
    memcpy(ptr, &bhdr, sizeof(bhdr));
    ptr += sizeof(bhdr);

    for (uint32_t v = client_topology_version + 1U; v <= server->topology_version; ++v) {
        uint32_t slot = (v - 1U) % HT_FED_SPLIT_LOG_CAP;
        memcpy(ptr, &server->split_log[slot], sizeof(ht_fed_split_cmd_t));
        ptr += sizeof(ht_fed_split_cmd_t);
    }

    return ht_fed_finalize_frame(out_buf, buf_cap, HT_FED_MSG_VFDT_SPLIT_BROADCAST,
                                 flags, target_client_id, server->round_id,
                                 server->topology_version, payload_len, out_written);
}
