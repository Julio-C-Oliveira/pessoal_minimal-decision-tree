/**
 * @file ht_fed_forest.c
 * @brief Implementation of Mode 1 Federated Forest (Fed-Forest) client and server.
 */

#include "ht_fed_forest.h"
#include <string.h>

typedef struct {
    ht_float_t reported_error;
    ht_float_t train_weight;
} ht_fed_forest_uplink_meta_t;

typedef struct {
    uint16_t n_trees;
    uint16_t reserved;
} ht_fed_forest_bundle_hdr_t;

typedef struct {
    uint16_t origin_client_id;
    uint16_t reserved;
    uint32_t tree_bytes;
    ht_float_t reported_error;
} ht_fed_forest_bundle_item_t;

/* ========================================================================= */
/* --- Client Implementation ----------------------------------------------- */
/* ========================================================================= */

void ht_fed_forest_default_client_config(ht_fed_forest_client_cfg_t *cfg,
                                         uint16_t client_id,
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
    cfg->voting = HT_VOTE_WEIGHTED;
    cfg->error_decay = HT_CONST(0.98);
    cfg->client_id = client_id;
}

ht_status_t ht_fed_forest_client_init(ht_fed_forest_client_t *client,
                                      const ht_fed_forest_client_cfg_t *cfg) {
    if (client == NULL || cfg == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    memset(client, 0, sizeof(*client));
    ht_status_t st = ht_tree_init(&client->local_tree, &cfg->tree_params);
    if (st != HT_OK) {
        return st;
    }
    client->voting = cfg->voting;
    client->error_decay = (cfg->error_decay > HT_CONST(0.0) && cfg->error_decay < HT_CONST(1.0))
        ? cfg->error_decay : HT_CONST(0.98);
    client->client_id = cfg->client_id;
    client->n_peers = 0U;
    client->round_id = 0U;

    ht_float_t init_err = (cfg->tree_params.task_type == HT_TASK_REGRESSION)
        ? HT_CONST(1.0) : HT_CONST(0.25);
    for (uint16_t i = 0; i < 1U + HT_FED_FOREST_MAX_TREES; ++i) {
        client->member_faded_error[i] = init_err;
    }
    return HT_OK;
}

ht_status_t ht_fed_forest_client_set_adwin_pool(ht_fed_forest_client_t *client,
                                                ht_adwin_pool_t *pool) {
    if (client == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    return ht_tree_set_adwin_pool(&client->local_tree, pool);
}

ht_status_t ht_fed_forest_client_predict_proba(const ht_fed_forest_client_t *client,
                                               const ht_value_t x[HT_MAX_FEATURES],
                                               ht_float_t proba_out[HT_MAX_CLASSES]) {
    if (client == NULL || x == NULL || proba_out == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t n_members = (uint16_t)(1U + client->n_peers);
    ht_float_t probas[(1U + HT_FED_FOREST_MAX_TREES) * HT_MAX_CLASSES];
    ht_float_t weights[1U + HT_FED_FOREST_MAX_TREES];

    ht_status_t st = ht_tree_predict_proba(&client->local_tree, x, &probas[0]);
    if (st != HT_OK) {
        return st;
    }
    weights[0] = ht_voting_member_weight(client->voting, HT_TASK_CLASSIFICATION,
                                         client->member_faded_error[0]);

    for (uint8_t p = 0; p < client->n_peers; ++p) {
        ht_float_t *pm = &probas[(size_t)(p + 1U) * HT_MAX_CLASSES];
        if (ht_compact_tree_predict_proba(&client->peer_trees[p], x, pm) == HT_OK) {
            weights[p + 1U] = ht_voting_member_weight(client->voting, HT_TASK_CLASSIFICATION,
                                                      client->member_faded_error[p + 1U]);
        } else {
            weights[p + 1U] = HT_CONST(0.0);
        }
    }

    return ht_voting_combine_proba(n_members, probas, weights,
                                   client->local_tree.params.n_classes, proba_out);
}

uint8_t ht_fed_forest_client_predict(const ht_fed_forest_client_t *client,
                                     const ht_value_t x[HT_MAX_FEATURES]) {
    ht_float_t probas[HT_MAX_CLASSES];
    if (ht_fed_forest_client_predict_proba(client, x, probas) != HT_OK) {
        return 0U;
    }
    uint8_t best_c = 0U;
    ht_float_t best_p = HT_CONST(-1.0);
    for (uint8_t c = 0; c < client->local_tree.params.n_classes; ++c) {
        if (probas[c] > best_p) {
            best_p = probas[c];
            best_c = c;
        }
    }
    return best_c;
}

ht_status_t ht_fed_forest_client_learn_one(ht_fed_forest_client_t *client,
                                           const ht_value_t x[HT_MAX_FEATURES],
                                           uint8_t y,
                                           ht_float_t weight) {
    if (client == NULL || x == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    /* Test-Then-Train: update prequential error for local tree and all peer trees */
    uint8_t local_pred = ht_tree_predict(&client->local_tree, x);
    ht_voting_update_error_classif(&client->member_faded_error[0], local_pred, y, client->error_decay);

    for (uint8_t p = 0; p < client->n_peers; ++p) {
        uint8_t peer_pred = ht_compact_tree_predict(&client->peer_trees[p], x);
        ht_voting_update_error_classif(&client->member_faded_error[p + 1U], peer_pred, y, client->error_decay);
    }

    /* Only local_tree learns incrementally */
    return ht_tree_learn_one(&client->local_tree, x, y, weight);
}

ht_float_t ht_fed_forest_client_predict_reg(const ht_fed_forest_client_t *client,
                                            const ht_value_t x[HT_MAX_FEATURES]) {
    if (client == NULL || x == NULL) {
        return HT_CONST(0.0);
    }

    uint16_t n_members = (uint16_t)(1U + client->n_peers);
    ht_float_t preds[1U + HT_FED_FOREST_MAX_TREES];
    ht_float_t weights[1U + HT_FED_FOREST_MAX_TREES];

    preds[0] = ht_tree_predict_reg(&client->local_tree, x);
    weights[0] = ht_voting_member_weight(client->voting, HT_TASK_REGRESSION,
                                         client->member_faded_error[0]);

    for (uint8_t p = 0; p < client->n_peers; ++p) {
        preds[p + 1U] = ht_compact_tree_predict_reg(&client->peer_trees[p], x);
        weights[p + 1U] = ht_voting_member_weight(client->voting, HT_TASK_REGRESSION,
                                                  client->member_faded_error[p + 1U]);
    }

    return ht_voting_combine_reg(n_members, preds, weights);
}

ht_status_t ht_fed_forest_client_learn_one_reg(ht_fed_forest_client_t *client,
                                               const ht_value_t x[HT_MAX_FEATURES],
                                               ht_float_t y,
                                               ht_float_t weight) {
    if (client == NULL || x == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    ht_float_t local_pred = ht_tree_predict_reg(&client->local_tree, x);
    ht_voting_update_error_reg(&client->member_faded_error[0], local_pred, y, client->error_decay);

    for (uint8_t p = 0; p < client->n_peers; ++p) {
        ht_float_t peer_pred = ht_compact_tree_predict_reg(&client->peer_trees[p], x);
        ht_voting_update_error_reg(&client->member_faded_error[p + 1U], peer_pred, y, client->error_decay);
    }

    return ht_tree_learn_one_reg(&client->local_tree, x, y, weight);
}

ht_status_t ht_fed_forest_client_export_update(const ht_fed_forest_client_t *client,
                                               void *out_buf,
                                               size_t buf_cap,
                                               size_t *out_written) {
    if (client == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    ht_compact_tree_t ct;
    ht_status_t st = ht_compact_tree_from_tree(&ct, &client->local_tree);
    if (st != HT_OK) {
        return st;
    }

    size_t ct_bytes = 0U;
    st = ht_compact_tree_save(&ct, NULL, 0U, &ct_bytes);
    if (st != HT_OK) {
        return st;
    }

    uint32_t payload_len = (uint32_t)(sizeof(ht_fed_forest_uplink_meta_t) + ct_bytes);
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

    uint8_t *payload_ptr = (uint8_t *)out_buf + sizeof(ht_fed_header_t);
    ht_fed_forest_uplink_meta_t meta;
    meta.reported_error = client->member_faded_error[0];
    meta.train_weight = client->local_tree.train_weight_seen;
    memcpy(payload_ptr, &meta, sizeof(meta));

    size_t ct_written = 0U;
    st = ht_compact_tree_save(&ct, payload_ptr + sizeof(meta), ct_bytes, &ct_written);
    if (st != HT_OK) {
        return st;
    }

    uint16_t flags = ht_fed_make_flags(client->local_tree.params.task_type);
    return ht_fed_finalize_frame(out_buf, buf_cap, HT_FED_MSG_FOREST_CLIENT_TREE,
                                 flags, client->client_id, client->round_id,
                                 0U, payload_len, out_written);
}

ht_status_t ht_fed_forest_client_import_forest(ht_fed_forest_client_t *client,
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
    if (hdr.msg_type != HT_FED_MSG_FOREST_SERVER_BUNDLE) {
        return HT_ERR_CORRUPT_IMAGE;
    }
    uint16_t expected_flags = ht_fed_make_flags(client->local_tree.params.task_type);
    if (hdr.flags != expected_flags) {
        return HT_ERR_CORRUPT_IMAGE;
    }
    if (hdr.payload_len < sizeof(ht_fed_forest_bundle_hdr_t)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    ht_fed_forest_bundle_hdr_t bhdr;
    memcpy(&bhdr, payload, sizeof(bhdr));
    size_t offset = sizeof(bhdr);

    uint8_t loaded = 0U;
    for (uint16_t i = 0; i < bhdr.n_trees; ++i) {
        if (offset + sizeof(ht_fed_forest_bundle_item_t) > hdr.payload_len) {
            return HT_ERR_CORRUPT_IMAGE;
        }
        ht_fed_forest_bundle_item_t item;
        memcpy(&item, payload + offset, sizeof(item));
        offset += sizeof(item);

        if (offset + item.tree_bytes > hdr.payload_len) {
            return HT_ERR_CORRUPT_IMAGE;
        }

        if (loaded < HT_FED_FOREST_MAX_TREES) {
            st = ht_compact_tree_load(&client->peer_trees[loaded],
                                      payload + offset, item.tree_bytes);
            if (st != HT_OK) {
                return st;
            }
            client->peer_client_ids[loaded] = item.origin_client_id;
            /* Initialize peer faded error using server-reported value (clamped) */
            ht_float_t err_init = (item.reported_error >= HT_CONST(0.0))
                ? item.reported_error : HT_CONST(0.25);
            client->member_faded_error[loaded + 1U] = err_init;
            loaded++;
        }
        offset += item.tree_bytes;
    }

    client->n_peers = loaded;
    client->round_id = hdr.round_id;
    return HT_OK;
}

/* ========================================================================= */
/* --- Server Implementation ----------------------------------------------- */
/* ========================================================================= */

void ht_fed_forest_default_server_config(ht_fed_forest_server_cfg_t *cfg,
                                         ht_task_type_t task_type) {
    if (cfg == NULL) {
        return;
    }
    cfg->task_type = task_type;
    cfg->agg_policy = HT_FED_FOREST_AGG_TOP_M;
    cfg->max_forest_size = (uint8_t)HT_FED_FOREST_MAX_TREES;
}

ht_status_t ht_fed_forest_server_init(ht_fed_forest_server_t *server,
                                      const ht_fed_forest_server_cfg_t *cfg) {
    if (server == NULL || cfg == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    memset(server, 0, sizeof(*server));
    server->cfg = *cfg;
    if (server->cfg.max_forest_size == 0U || server->cfg.max_forest_size > HT_FED_FOREST_MAX_TREES) {
        server->cfg.max_forest_size = (uint8_t)HT_FED_FOREST_MAX_TREES;
    }
    return HT_OK;
}

ht_status_t ht_fed_forest_server_ingest(ht_fed_forest_server_t *server,
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
    if (hdr.msg_type != HT_FED_MSG_FOREST_CLIENT_TREE) {
        return HT_ERR_CORRUPT_IMAGE;
    }
    if (hdr.flags != ht_fed_make_flags(server->cfg.task_type)) {
        return HT_ERR_CORRUPT_IMAGE;
    }
    if (hdr.payload_len <= sizeof(ht_fed_forest_uplink_meta_t)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    ht_fed_forest_uplink_meta_t meta;
    memcpy(&meta, payload, sizeof(meta));

    /* Find existing slot for client_id, or first empty slot, or evict worst/oldest */
    uint16_t slot = HT_INVALID_IDX;
    for (uint16_t i = 0; i < HT_FED_SERVER_MAX_TREES; ++i) {
        if (server->entries[i].occupied && server->entries[i].client_id == hdr.client_id) {
            slot = i;
            break;
        }
    }
    if (slot == HT_INVALID_IDX) {
        for (uint16_t i = 0; i < HT_FED_SERVER_MAX_TREES; ++i) {
            if (!server->entries[i].occupied) {
                slot = i;
                break;
            }
        }
    }
    if (slot == HT_INVALID_IDX) {
        /* Evict entry with worst reported error (tie-break: oldest round) */
        slot = 0U;
        for (uint16_t i = 1U; i < HT_FED_SERVER_MAX_TREES; ++i) {
            if (server->entries[i].reported_error > server->entries[slot].reported_error ||
                (server->entries[i].reported_error == server->entries[slot].reported_error &&
                 server->entries[i].last_round < server->entries[slot].last_round)) {
                slot = i;
            }
        }
    }

    ht_compact_tree_t parsed;
    st = ht_compact_tree_load(&parsed,
                              payload + sizeof(meta),
                              hdr.payload_len - sizeof(meta));
    if (st != HT_OK) {
        return st;
    }

    bool was_occupied = (server->entries[slot].occupied != 0U);
    server->entries[slot].tree = parsed;
    server->entries[slot].client_id = hdr.client_id;
    server->entries[slot].reported_error = meta.reported_error;
    server->entries[slot].train_weight = meta.train_weight;
    server->entries[slot].last_round = server->round_id;
    server->entries[slot].occupied = 1U;
    if (!was_occupied) {
        server->n_entries++;
    }

    return HT_OK;
}

ht_status_t ht_fed_forest_server_build_forest(ht_fed_forest_server_t *server,
                                              uint16_t target_client_id,
                                              bool exclude_self,
                                              void *out_buf,
                                              size_t buf_cap,
                                              size_t *out_written) {
    if (server == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t candidates[HT_FED_SERVER_MAX_TREES];
    uint16_t n_cand = 0U;

    for (uint16_t i = 0; i < HT_FED_SERVER_MAX_TREES; ++i) {
        if (!server->entries[i].occupied) {
            continue;
        }
        if (exclude_self && server->entries[i].client_id == target_client_id) {
            continue;
        }
        candidates[n_cand++] = i;
    }

    uint16_t selected[HT_FED_FOREST_MAX_TREES];
    uint16_t n_sel = 0U;
    uint16_t limit = server->cfg.max_forest_size;
    if (limit > HT_FED_FOREST_MAX_TREES) {
        limit = (uint16_t)HT_FED_FOREST_MAX_TREES;
    }

    if (n_cand > 0U) {
        if (server->cfg.agg_policy == HT_FED_FOREST_AGG_TOP_M) {
            /* Deterministic insertion sort by lowest reported_error, tie-break highest train_weight */
            for (uint16_t i = 1U; i < n_cand; ++i) {
                uint16_t key = candidates[i];
                int32_t j = (int32_t)i - 1;
                while (j >= 0) {
                    const ht_fed_forest_entry_t *ej = &server->entries[candidates[j]];
                    const ht_fed_forest_entry_t *ek = &server->entries[key];
                    bool k_better = (ek->reported_error < ej->reported_error) ||
                                    (ek->reported_error == ej->reported_error && ek->train_weight > ej->train_weight);
                    if (!k_better) {
                        break;
                    }
                    candidates[j + 1] = candidates[j];
                    j--;
                }
                candidates[j + 1] = key;
            }
            n_sel = (n_cand < limit) ? n_cand : limit;
            for (uint16_t i = 0; i < n_sel; ++i) {
                selected[i] = candidates[i];
            }
        } else if (server->cfg.agg_policy == HT_FED_FOREST_AGG_ROUND_ROBIN) {
            n_sel = (n_cand < limit) ? n_cand : limit;
            uint16_t start = server->rr_cursor % n_cand;
            for (uint16_t i = 0; i < n_sel; ++i) {
                selected[i] = candidates[(start + i) % n_cand];
            }
        } else { /* HT_FED_FOREST_AGG_ALL */
            n_sel = (n_cand < limit) ? n_cand : limit;
            for (uint16_t i = 0; i < n_sel; ++i) {
                selected[i] = candidates[i];
            }
        }
    }

    /* Compute exact payload size */
    size_t payload_len = sizeof(ht_fed_forest_bundle_hdr_t);
    size_t item_sizes[HT_FED_FOREST_MAX_TREES];
    for (uint16_t i = 0; i < n_sel; ++i) {
        size_t ct_bytes = 0U;
        ht_status_t st = ht_compact_tree_save(&server->entries[selected[i]].tree, NULL, 0U, &ct_bytes);
        if (st != HT_OK) {
            return st;
        }
        item_sizes[i] = ct_bytes;
        payload_len += sizeof(ht_fed_forest_bundle_item_t) + ct_bytes;
    }

    size_t total = sizeof(ht_fed_header_t) + payload_len;
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
    ht_fed_forest_bundle_hdr_t bhdr;
    bhdr.n_trees = n_sel;
    bhdr.reserved = 0U;
    memcpy(ptr, &bhdr, sizeof(bhdr));
    ptr += sizeof(bhdr);

    for (uint16_t i = 0; i < n_sel; ++i) {
        const ht_fed_forest_entry_t *e = &server->entries[selected[i]];
        ht_fed_forest_bundle_item_t item;
        item.origin_client_id = e->client_id;
        item.reserved = 0U;
        item.tree_bytes = (uint32_t)item_sizes[i];
        item.reported_error = e->reported_error;
        memcpy(ptr, &item, sizeof(item));
        ptr += sizeof(item);

        size_t written_ct = 0U;
        ht_status_t st = ht_compact_tree_save(&e->tree, ptr, item_sizes[i], &written_ct);
        if (st != HT_OK) {
            return st;
        }
        ptr += written_ct;
    }

    uint16_t flags = ht_fed_make_flags(server->cfg.task_type);
    return ht_fed_finalize_frame(out_buf, buf_cap, HT_FED_MSG_FOREST_SERVER_BUNDLE,
                                 flags, target_client_id, server->round_id,
                                 0U, (uint32_t)payload_len, out_written);
}

void ht_fed_forest_server_advance_round(ht_fed_forest_server_t *server) {
    if (server == NULL) {
        return;
    }
    server->round_id++;
    if (server->n_entries > 0U) {
        server->rr_cursor = (uint16_t)((server->rr_cursor + 1U) % server->n_entries);
    }
}
