/**
 * @file ht_compact_tree.c
 * @brief Implementation of BFS-compacted inference-only decision tree.
 */

#include "ht_compact_tree.h"
#include <string.h>

static void ht_summarize_tree_subtree(const ht_tree_t *src,
                                      uint16_t start_idx,
                                      ht_float_t out_class_counts[HT_MAX_CLASSES],
                                      ht_var_stat_t *out_reg_stats) {
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        out_class_counts[c] = (ht_float_t)0.0;
    }
    ht_var_init(out_reg_stats);

    uint16_t stack[128];
    uint16_t top = 0;
    stack[top++] = start_idx;

    while (top > 0) {
        uint16_t curr = stack[--top];
        if (curr == HT_INVALID_IDX || curr >= HT_MAX_NODES) {
            continue;
        }
        const ht_node_t *n = &src->memory.node_pool[curr];
        if (!(n->flags & HT_NODE_IN_USE)) {
            continue;
        }
        if (n->flags & HT_NODE_IS_LEAF) {
            if (src->params.task_type == HT_TASK_REGRESSION) {
                *out_reg_stats = ht_var_combine(out_reg_stats, &n->reg_stats);
            } else {
                for (uint8_t c = 0; c < src->params.n_classes; ++c) {
                    out_class_counts[c] += n->class_counts[c];
                }
            }
        } else {
            if (n->left_child != HT_INVALID_IDX && top < 128U) {
                stack[top++] = n->left_child;
            }
            if (n->right_child != HT_INVALID_IDX && top < 128U) {
                stack[top++] = n->right_child;
            }
        }
    }
}

static ht_float_t ht_compact_subtree_weight(const ht_compact_tree_t *ct, uint16_t node_idx) {
    if (node_idx == HT_INVALID_IDX || node_idx >= ct->num_nodes) {
        return (ht_float_t)0.0;
    }
    ht_float_t total = (ht_float_t)0.0;
    uint16_t stack[64];
    uint16_t top = 0;
    stack[top++] = node_idx;

    while (top > 0) {
        uint16_t curr = stack[--top];
        if (curr >= ct->num_nodes) {
            continue;
        }
        const ht_node_t *n = &ct->nodes[curr];
        if (!(n->flags & HT_NODE_IN_USE)) {
            continue;
        }
        if (n->flags & HT_NODE_IS_LEAF) {
            if (ct->task_type == HT_TASK_REGRESSION) {
                total += n->reg_stats.count;
            } else {
                for (uint8_t c = 0; c < ct->n_classes; ++c) {
                    total += n->class_counts[c];
                }
            }
        } else {
            if (n->left_child != HT_INVALID_IDX && top < 64U) {
                stack[top++] = n->left_child;
            }
            if (n->right_child != HT_INVALID_IDX && top < 64U) {
                stack[top++] = n->right_child;
            }
        }
    }
    return total;
}

static uint16_t ht_compact_most_common_child(const ht_compact_tree_t *ct, const ht_node_t *node) {
    ht_float_t left_w = ht_compact_subtree_weight(ct, node->left_child);
    ht_float_t right_w = ht_compact_subtree_weight(ct, node->right_child);
    return (left_w >= right_w) ? node->left_child : node->right_child;
}

ht_status_t ht_compact_tree_from_tree(ht_compact_tree_t *dst, const ht_tree_t *src) {
    if (dst == NULL || src == NULL || src->root_idx == HT_INVALID_IDX || src->root_idx >= HT_MAX_NODES) {
        return HT_ERR_INVALID_ARG;
    }

    memset(dst, 0, sizeof(*dst));
    dst->task_type = src->params.task_type;
    dst->n_features = src->params.n_features;
    dst->n_classes = src->params.n_classes;
    dst->train_weight_seen = src->train_weight_seen;
    dst->has_subspace = 0U;
    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        dst->feature_types[f] = src->params.feature_types[f];
        dst->feature_map[f] = f;
    }

    uint16_t bfs_src[HT_COMPACT_MAX_NODES];
    uint16_t head = 0U;
    uint16_t tail = 0U;

    bfs_src[tail++] = src->root_idx;
    dst->root_idx = 0U;

    while (head < tail) {
        uint16_t s_idx = bfs_src[head];
        uint16_t d_idx = head;
        head++;

        const ht_node_t *sn = &src->memory.node_pool[s_idx];
        ht_node_t *dn = &dst->nodes[d_idx];
        memset(dn, 0, sizeof(*dn));

        if ((sn->flags & HT_NODE_IS_LEAF) ||
            sn->left_child == HT_INVALID_IDX ||
            sn->right_child == HT_INVALID_IDX) {
            dn->flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
            dn->stats_idx = HT_INVALID_IDX;
            if (src->params.task_type == HT_TASK_REGRESSION) {
                dn->reg_stats = sn->reg_stats;
            } else {
                for (uint8_t c = 0; c < src->params.n_classes; ++c) {
                    dn->class_counts[c] = sn->class_counts[c];
                }
            }
        } else if ((uint32_t)tail + 2U <= (uint32_t)HT_COMPACT_MAX_NODES) {
            uint16_t l_dst = tail++;
            uint16_t r_dst = tail++;
            bfs_src[l_dst] = sn->left_child;
            bfs_src[r_dst] = sn->right_child;

            dn->flags = (uint8_t)((sn->flags & HT_NODE_SPLIT_CAT) | HT_NODE_IN_USE);
            dn->feature_idx = sn->feature_idx;
            dn->split_val = sn->split_val;
            dn->left_child = l_dst;
            dn->right_child = r_dst;
        } else {
            /* Graceful capacity truncation: summarize remaining subtree into a frontier leaf */
            ht_float_t cc[HT_MAX_CLASSES];
            ht_var_stat_t rs;
            ht_summarize_tree_subtree(src, s_idx, cc, &rs);
            dn->flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
            dn->stats_idx = HT_INVALID_IDX;
            if (src->params.task_type == HT_TASK_REGRESSION) {
                dn->reg_stats = rs;
            } else {
                for (uint8_t c = 0; c < src->params.n_classes; ++c) {
                    dn->class_counts[c] = cc[c];
                }
            }
        }
    }

    dst->num_nodes = tail;
    return HT_OK;
}

ht_status_t ht_compact_tree_set_subspace(ht_compact_tree_t *ct,
                                         const uint8_t *feature_map,
                                         uint8_t n_subspace_features) {
    if (ct == NULL || feature_map == NULL || n_subspace_features == 0U ||
        n_subspace_features > HT_MAX_FEATURES) {
        return HT_ERR_INVALID_ARG;
    }
    ct->has_subspace = 1U;
    ct->n_features = n_subspace_features;
    for (uint8_t f = 0; f < n_subspace_features; ++f) {
        ct->feature_map[f] = feature_map[f];
    }
    return HT_OK;
}

uint16_t ht_compact_tree_route(const ht_compact_tree_t *ct,
                               const ht_value_t x[HT_MAX_FEATURES]) {
    if (ct == NULL || x == NULL || ct->num_nodes == 0U || ct->root_idx >= ct->num_nodes) {
        return HT_INVALID_IDX;
    }

    uint16_t curr = ct->root_idx;
    uint16_t guard = 0U;

    while (curr != HT_INVALID_IDX && curr < ct->num_nodes && guard < ct->num_nodes) {
        const ht_node_t *node = &ct->nodes[curr];
        if (node->flags & HT_NODE_IS_LEAF) {
            return curr;
        }

        uint8_t f_local = node->feature_idx;
        if (f_local >= HT_MAX_FEATURES) {
            return HT_INVALID_IDX;
        }
        uint8_t f_global = ct->has_subspace ? ct->feature_map[f_local] : f_local;
        if (f_global >= HT_MAX_FEATURES) {
            return HT_INVALID_IDX;
        }

        uint16_t next_idx;
        if (node->flags & HT_NODE_SPLIT_CAT) {
            int32_t cat = x[f_global].cat;
            if (cat < 0 || cat >= (int32_t)HT_MAX_CATEGORIES) {
                next_idx = ht_compact_most_common_child(ct, node);
            } else if (cat == node->split_val.cat_val) {
                next_idx = node->left_child;
            } else {
                next_idx = node->right_child;
            }
        } else {
            ht_float_t num = x[f_global].num;
            if (HT_ISNAN(num)) {
                next_idx = ht_compact_most_common_child(ct, node);
            } else if (num <= node->split_val.threshold) {
                next_idx = node->left_child;
            } else {
                next_idx = node->right_child;
            }
        }

        curr = next_idx;
        guard++;
    }

    return HT_INVALID_IDX;
}

ht_status_t ht_compact_tree_predict_proba(const ht_compact_tree_t *ct,
                                          const ht_value_t x[HT_MAX_FEATURES],
                                          ht_float_t proba_out[HT_MAX_CLASSES]) {
    if (ct == NULL || x == NULL || proba_out == NULL || ct->n_classes == 0U) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t leaf_idx = ht_compact_tree_route(ct, x);
    if (leaf_idx == HT_INVALID_IDX) {
        return HT_ERR_INVALID_ARG;
    }

    const ht_node_t *leaf = &ct->nodes[leaf_idx];
    ht_float_t total = (ht_float_t)0.0;
    for (uint8_t c = 0; c < ct->n_classes; ++c) {
        total += leaf->class_counts[c];
    }

    if (total <= (ht_float_t)0.0) {
        ht_float_t uni = (ht_float_t)1.0 / (ht_float_t)ct->n_classes;
        for (uint8_t c = 0; c < ct->n_classes; ++c) {
            proba_out[c] = uni;
        }
    } else {
        for (uint8_t c = 0; c < ct->n_classes; ++c) {
            proba_out[c] = leaf->class_counts[c] / total;
        }
    }
    for (uint8_t c = ct->n_classes; c < HT_MAX_CLASSES; ++c) {
        proba_out[c] = (ht_float_t)0.0;
    }
    return HT_OK;
}

uint8_t ht_compact_tree_predict(const ht_compact_tree_t *ct,
                                const ht_value_t x[HT_MAX_FEATURES]) {
    ht_float_t probas[HT_MAX_CLASSES];
    if (ht_compact_tree_predict_proba(ct, x, probas) != HT_OK) {
        return 0U;
    }
    uint8_t best_c = 0U;
    ht_float_t best_p = (ht_float_t)-1.0;
    for (uint8_t c = 0; c < ct->n_classes; ++c) {
        if (probas[c] > best_p) {
            best_p = probas[c];
            best_c = c;
        }
    }
    return best_c;
}

ht_float_t ht_compact_tree_predict_reg(const ht_compact_tree_t *ct,
                                       const ht_value_t x[HT_MAX_FEATURES]) {
    if (ct == NULL || x == NULL) {
        return (ht_float_t)0.0;
    }
    uint16_t leaf_idx = ht_compact_tree_route(ct, x);
    if (leaf_idx == HT_INVALID_IDX) {
        return (ht_float_t)0.0;
    }
    return ct->nodes[leaf_idx].reg_stats.mean;
}

ht_status_t ht_compact_tree_save(const ht_compact_tree_t *ct,
                                 void *buffer,
                                 size_t buffer_cap,
                                 size_t *out_bytes_written) {
    if (ct == NULL || ct->num_nodes == 0U || ct->num_nodes > HT_COMPACT_MAX_NODES) {
        return HT_ERR_INVALID_ARG;
    }

    size_t total_size = sizeof(ht_compact_serial_header_t) +
                        (size_t)ct->num_nodes * sizeof(ht_node_t);
    if (out_bytes_written != NULL) {
        *out_bytes_written = total_size;
    }
    if (buffer == NULL) {
        return HT_OK;
    }
    if (buffer_cap < total_size) {
        return HT_ERR_BUFFER_TOO_SMALL;
    }

    ht_compact_serial_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = HT_COMPACT_MAGIC;
    hdr.version = HT_COMPACT_VERSION;
    hdr.flags = 0U;
#ifdef HT_USE_DOUBLE
    hdr.flags |= HT_COMPACT_FLAG_DOUBLE;
#endif
    if (ct->task_type == HT_TASK_REGRESSION) {
        hdr.flags |= HT_COMPACT_FLAG_REG;
    }
    if (ct->has_subspace) {
        hdr.flags |= HT_COMPACT_FLAG_SUBSPC;
    }
    hdr.float_size = (uint16_t)sizeof(ht_float_t);
    hdr.num_nodes = ct->num_nodes;
    hdr.root_idx = ct->root_idx;
    hdr.n_features = ct->n_features;
    hdr.n_classes = ct->n_classes;
    hdr.train_weight_seen = ct->train_weight_seen;
    memcpy(hdr.feature_types, ct->feature_types, sizeof(hdr.feature_types));
    memcpy(hdr.feature_map, ct->feature_map, sizeof(hdr.feature_map));

    uint8_t *ptr = (uint8_t *)buffer;
    memcpy(ptr, &hdr, sizeof(hdr));
    ptr += sizeof(hdr);
    memcpy(ptr, ct->nodes, (size_t)ct->num_nodes * sizeof(ht_node_t));

    return HT_OK;
}

ht_status_t ht_compact_tree_load(ht_compact_tree_t *ct,
                                 const void *buffer,
                                 size_t buffer_len) {
    if (ct == NULL || buffer == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    if (buffer_len < sizeof(ht_compact_serial_header_t)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    const uint8_t *ptr = (const uint8_t *)buffer;
    ht_compact_serial_header_t hdr;
    memcpy(&hdr, ptr, sizeof(hdr));
    ptr += sizeof(hdr);

    if (hdr.magic != HT_COMPACT_MAGIC ||
        hdr.version != HT_COMPACT_VERSION ||
        hdr.float_size != (uint16_t)sizeof(ht_float_t) ||
        hdr.num_nodes == 0U ||
        hdr.num_nodes > HT_COMPACT_MAX_NODES ||
        hdr.root_idx >= hdr.num_nodes ||
        hdr.n_features == 0U ||
        hdr.n_features > HT_MAX_FEATURES) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    size_t expected = sizeof(ht_compact_serial_header_t) +
                      (size_t)hdr.num_nodes * sizeof(ht_node_t);
    if (buffer_len < expected) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    memset(ct, 0, sizeof(*ct));
    ct->task_type = (hdr.flags & HT_COMPACT_FLAG_REG) ? HT_TASK_REGRESSION : HT_TASK_CLASSIFICATION;
    ct->has_subspace = (hdr.flags & HT_COMPACT_FLAG_SUBSPC) ? 1U : 0U;
    ct->num_nodes = hdr.num_nodes;
    ct->root_idx = hdr.root_idx;
    ct->n_features = hdr.n_features;
    ct->n_classes = hdr.n_classes;
    ct->train_weight_seen = hdr.train_weight_seen;
    memcpy(ct->feature_types, hdr.feature_types, sizeof(ct->feature_types));
    memcpy(ct->feature_map, hdr.feature_map, sizeof(ct->feature_map));
    memcpy(ct->nodes, ptr, (size_t)hdr.num_nodes * sizeof(ht_node_t));

    /* Basic structural bounds check on loaded branches */
    for (uint16_t i = 0; i < ct->num_nodes; ++i) {
        if (!(ct->nodes[i].flags & HT_NODE_IS_LEAF)) {
            if (ct->nodes[i].left_child >= ct->num_nodes ||
                ct->nodes[i].right_child >= ct->num_nodes) {
                return HT_ERR_CORRUPT_IMAGE;
            }
        }
    }

    return HT_OK;
}
