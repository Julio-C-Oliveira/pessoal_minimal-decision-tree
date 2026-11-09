/**
 * @file ht_tree.c
 * @brief Complete Hoeffding Tree (VFDT) implementation in pure C11.
 *
 * Implements deterministic routing, multi-strategy leaf predictions (MC, NB, NBA),
 * online learning, memory-bounded leaf deactivation, and binary serialization.
 */

#include "ht_tree.h"
#include "ht_prep.h"
#include <string.h>

/* ========================================================================= */
/* --- Internal Static Helpers --------------------------------------------- */
/* ========================================================================= */

static uint16_t ht_most_common_child(const ht_tree_t *tree, const ht_node_t *node) {
    ht_float_t left_w = (ht_float_t)0.0;
    ht_float_t right_w = (ht_float_t)0.0;

    if (node->left_child != HT_INVALID_IDX) {
        const ht_node_t *left = &tree->memory.node_pool[node->left_child];
        if (tree->params.task_type == HT_TASK_REGRESSION) {
            left_w = left->reg_stats.count;
        } else {
            for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
                left_w += left->class_counts[c];
            }
        }
    }
    if (node->right_child != HT_INVALID_IDX) {
        const ht_node_t *right = &tree->memory.node_pool[node->right_child];
        if (tree->params.task_type == HT_TASK_REGRESSION) {
            right_w = right->reg_stats.count;
        } else {
            for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
                right_w += right->class_counts[c];
            }
        }
    }

    return (left_w >= right_w) ? node->left_child : node->right_child;
}

static ht_float_t ht_tree_normalize_feature_for_sgd(const ht_tree_t *tree,
                                                    const ht_leaf_stats_t *stats,
                                                    uint8_t f,
                                                    ht_float_t val) {
    /* 1. Global tree-level prior */
    ht_float_t mu_tree = (ht_float_t)0.0;
    ht_float_t sigma2_tree = (ht_float_t)1.0;
    if (tree->feat_stats[f].count > (ht_float_t)1.0) {
        mu_tree = tree->feat_stats[f].mean;
        sigma2_tree = tree->feat_stats[f].m2 / tree->feat_stats[f].count;
        if (sigma2_tree < (ht_float_t)1e-6) {
            sigma2_tree = (ht_float_t)1e-6;
        }
    }

    /* 2. Local leaf statistics from the 5 slots of reg_num */
    ht_float_t n_slots = (ht_float_t)0.0;
    ht_float_t sum_prod = (ht_float_t)0.0;
    for (uint8_t s = 0; s < HT_REG_NUM_SLOTS; ++s) {
        ht_float_t w = stats->observers[f].reg_num.slots[s].y_stats.count;
        n_slots += w;
        sum_prod += w * stats->observers[f].reg_num.slots[s].x_mean;
    }

    ht_float_t mu_slots = mu_tree;
    ht_float_t sigma2_slots = sigma2_tree;
    if (n_slots > (ht_float_t)0.0) {
        mu_slots = sum_prod / n_slots;
        /* Sample variance across slot centroids */
        ht_float_t sum_sq_diff = (ht_float_t)0.0;
        for (uint8_t s = 0; s < HT_REG_NUM_SLOTS; ++s) {
            ht_float_t w = stats->observers[f].reg_num.slots[s].y_stats.count;
            if (w > (ht_float_t)0.0) {
                ht_float_t diff = stats->observers[f].reg_num.slots[s].x_mean - mu_slots;
                sum_sq_diff += w * diff * diff;
            }
        }
        sigma2_slots = sum_sq_diff / n_slots;
        if (sigma2_slots < (ht_float_t)1e-6) {
            sigma2_slots = sigma2_tree;
        }
    }

    /* 3. Credibility blend: alpha = n_slots / (n_slots + 10.0) */
    ht_float_t alpha = n_slots / (n_slots + (ht_float_t)10.0);
    ht_float_t mu_eff = alpha * mu_slots + ((ht_float_t)1.0 - alpha) * mu_tree;
    ht_float_t sigma2_eff = alpha * sigma2_slots + ((ht_float_t)1.0 - alpha) * sigma2_tree;
    if (sigma2_eff < (ht_float_t)1e-6) {
        sigma2_eff = (ht_float_t)1e-6;
    }

    return (val - mu_eff) / HT_SQRT(sigma2_eff);
}

static ht_float_t ht_leaf_predict_linear_model(const ht_tree_t *tree,
                                               const ht_leaf_stats_t *stats,
                                               const ht_value_t x[HT_MAX_FEATURES]) {
    ht_float_t pred = stats->reg_intercept;
    for (uint8_t f = 0; f < tree->params.n_features; ++f) {
        if (tree->params.feature_types[f] == HT_FEAT_NUMERIC) {
            if (!HT_ISNAN(x[f].num)) {
                ht_float_t val = x[f].num;
                if (tree->params.leaf_sgd_normalize) {
                    val = ht_tree_normalize_feature_for_sgd(tree, stats, f, val);
                }
                pred += stats->reg_weights[f] * val;
            }
        }
    }
    return pred;
}

static ht_float_t ht_predict_reg_at_leaf(const ht_tree_t *tree,
                                        uint16_t leaf_idx,
                                        const ht_value_t x[HT_MAX_FEATURES]) {
    if (tree == NULL || leaf_idx == HT_INVALID_IDX) {
        return (ht_float_t)0.0;
    }

    const ht_node_t *node = &tree->memory.node_pool[leaf_idx];
    if (node->stats_idx == HT_INVALID_IDX) {
        /* Inactive leaf fallback: mean target */
        return node->reg_stats.mean;
    }

    const ht_leaf_stats_t *stats = &tree->memory.stats_pool[node->stats_idx];

    if (tree->params.leaf_prediction == HT_LEAF_MEAN) {
        return node->reg_stats.mean;
    } else if (tree->params.leaf_prediction == HT_LEAF_MODEL) {
        return ht_leaf_predict_linear_model(tree, stats, x);
    } else { /* HT_LEAF_ADAPTIVE */
        if (stats->fmse_mean < stats->fmse_model) {
            return node->reg_stats.mean;
        } else {
            return ht_leaf_predict_linear_model(tree, stats, x);
        }
    }
}

static uint8_t ht_leaf_predict_mc_class(const ht_node_t *node, uint8_t n_classes) {
    uint8_t best_c = 0;
    ht_float_t best_w = (ht_float_t)-1.0;
    for (uint8_t c = 0; c < n_classes; ++c) {
        if (node->class_counts[c] > best_w) {
            best_w = node->class_counts[c];
            best_c = c;
        }
    }
    return best_c;
}

static void ht_leaf_predict_mc_proba(const ht_node_t *node,
                                     uint8_t n_classes,
                                     ht_float_t proba_out[HT_MAX_CLASSES]) {
    ht_float_t total = (ht_float_t)0.0;
    for (uint8_t c = 0; c < n_classes; ++c) {
        total += node->class_counts[c];
    }

    if (total <= (ht_float_t)0.0) {
        ht_float_t uniform = ((ht_float_t)1.0) / ((ht_float_t)n_classes);
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_out[c] = uniform;
        }
    } else {
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_out[c] = node->class_counts[c] / total;
        }
    }

    for (uint8_t c = n_classes; c < HT_MAX_CLASSES; ++c) {
        proba_out[c] = (ht_float_t)0.0;
    }
}

static bool ht_leaf_predict_nb_proba(const ht_tree_t *tree,
                                     const ht_node_t *node,
                                     const ht_leaf_stats_t *stats,
                                     const ht_value_t x[HT_MAX_FEATURES],
                                     ht_float_t proba_out[HT_MAX_CLASSES]) {
    uint8_t n_classes = tree->params.n_classes;
    uint8_t n_features = tree->params.n_features;

    ht_float_t total_weight = (ht_float_t)0.0;
    for (uint8_t c = 0; c < n_classes; ++c) {
        total_weight += node->class_counts[c];
    }

    if (total_weight <= (ht_float_t)0.0) {
        return false;
    }

    ht_float_t log_likelihood[HT_MAX_CLASSES];
    bool class_observed[HT_MAX_CLASSES];
    uint8_t num_observed = 0;
    ht_float_t max_ll = -HT_INFINITY;

    for (uint8_t c = 0; c < n_classes; ++c) {
        if (node->class_counts[c] > (ht_float_t)0.0) {
            class_observed[c] = true;
            num_observed++;
            log_likelihood[c] = HT_LOG(node->class_counts[c] / total_weight);

            for (uint8_t f = 0; f < n_features; ++f) {
                if (tree->params.feature_types[f] == HT_FEAT_NUMERIC) {
                    if (!HT_ISNAN(x[f].num)) {
                        ht_float_t log_p;
                        if (tree->params.numeric_obs_type == HT_OBS_HISTOGRAM) {
                            log_p = ht_hist_cond_log_proba(&stats->observers[f].hist, c, x[f].num);
                        } else {
                            log_p = ht_gaussian_cond_log_proba(&stats->observers[f].num[c], x[f].num);
                        }
                        if (log_p > -HT_INFINITY) {
                            log_likelihood[c] += log_p;
                        }
                    }
                } else {
                    int32_t cat = x[f].cat;
                    if (cat >= 0 && cat < (int32_t)HT_MAX_CATEGORIES) {
                        ht_float_t tmp = ht_categorical_cond_proba(&stats->observers[f].cat, c, cat);
                        if (tmp > (ht_float_t)0.0) {
                            log_likelihood[c] += HT_LOG(tmp);
                        }
                    }
                }
            }

            if (log_likelihood[c] > max_ll) {
                max_ll = log_likelihood[c];
            }
        } else {
            class_observed[c] = false;
            log_likelihood[c] = -HT_INFINITY;
        }
    }

    if (num_observed == 0 || max_ll == -HT_INFINITY) {
        return false;
    }

    /* Log-Sum-Exp Trick */
    ht_float_t sum_exp = (ht_float_t)0.0;
    for (uint8_t c = 0; c < n_classes; ++c) {
        if (class_observed[c]) {
            sum_exp += HT_EXP(log_likelihood[c] - max_ll);
        }
    }

    if (sum_exp <= (ht_float_t)0.0) {
        return false;
    }

    ht_float_t lse = max_ll + HT_LOG(sum_exp);

    for (uint8_t c = 0; c < n_classes; ++c) {
        if (class_observed[c]) {
            proba_out[c] = HT_EXP(log_likelihood[c] - lse);
        } else {
            proba_out[c] = (ht_float_t)0.0;
        }
    }
    for (uint8_t c = n_classes; c < HT_MAX_CLASSES; ++c) {
        proba_out[c] = (ht_float_t)0.0;
    }

    return true;
}

static bool ht_leaf_predict_nb_class(const ht_tree_t *tree,
                                     const ht_node_t *node,
                                     const ht_leaf_stats_t *stats,
                                     const ht_value_t x[HT_MAX_FEATURES],
                                     uint8_t *out_class) {
    ht_float_t probas[HT_MAX_CLASSES];
    if (!ht_leaf_predict_nb_proba(tree, node, stats, x, probas)) {
        return false;
    }

    uint8_t best_c = 0;
    ht_float_t best_p = (ht_float_t)-1.0;
    for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
        if (probas[c] > best_p) {
            best_p = probas[c];
            best_c = c;
        }
    }

    *out_class = best_c;
    return true;
}

static void ht_predict_proba_at_leaf(const ht_tree_t *tree,
                                     uint16_t leaf_idx,
                                     const ht_value_t x[HT_MAX_FEATURES],
                                     ht_float_t proba_out[HT_MAX_CLASSES])
{
    const ht_node_t *node = &tree->memory.node_pool[leaf_idx];
    bool is_active = (node->stats_idx != HT_INVALID_IDX);

    if (is_active) {
        const ht_leaf_stats_t *stats = &tree->memory.stats_pool[node->stats_idx];

        if (tree->params.leaf_prediction == HT_LEAF_NB) {
            ht_float_t total = (ht_float_t)0.0;
            for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
                total += node->class_counts[c];
            }
            if (total >= tree->params.nb_threshold) {
                if (ht_leaf_predict_nb_proba(tree, node, stats, x, proba_out)) {
                    return;
                }
            }
        } else if (tree->params.leaf_prediction == HT_LEAF_NBA) {
            if (stats->nb_correct_weight >= stats->mc_correct_weight) {
                if (ht_leaf_predict_nb_proba(tree, node, stats, x, proba_out)) {
                    return;
                }
            }
        }
    }

    /* Fallback to Majority Class */
    ht_leaf_predict_mc_proba(node, tree->params.n_classes, proba_out);
}

static ht_status_t ht_tree_deactivate_least_promising_leaf_except(ht_tree_t *tree,
                                                                 uint16_t except_idx) {
    if (tree == NULL || tree->n_active_leaves == 0) {
        return HT_ERR_OUT_OF_STATS;
    }

    uint16_t min_idx = HT_INVALID_IDX;
    ht_float_t min_promise = HT_INFINITY;

    if (tree->params.task_type == HT_TASK_REGRESSION) {
        /* In regression: Promise(l) = -depth(l). Deepest leaf deactivated first. */
        uint16_t node_stack[64];
        uint16_t depth_stack[64];
        int top = 0;
        if (tree->root_idx != HT_INVALID_IDX) {
            node_stack[0] = tree->root_idx;
            depth_stack[0] = 0;
            top = 1;
        }

        while (top > 0) {
            top--;
            uint16_t curr = node_stack[top];
            uint16_t d = depth_stack[top];
            if (curr >= HT_MAX_NODES) continue;
            const ht_node_t *n = &tree->memory.node_pool[curr];
            if (!(n->flags & HT_NODE_IN_USE)) continue;

            if (n->flags & HT_NODE_IS_LEAF) {
                if (curr != except_idx && n->stats_idx != HT_INVALID_IDX) {
                    ht_float_t p = -(ht_float_t)d;
                    if (p < min_promise) {
                        min_promise = p;
                        min_idx = curr;
                    }
                }
            } else {
                if (tree->params.enable_hat && tree->adwin_pool != NULL) {
                    uint16_t alt = tree->node_drift[curr].alt_tree_idx;
                    if (alt != HT_INVALID_IDX && top < 62) {
                        node_stack[top] = alt;
                        depth_stack[top] = d;
                        top++;
                    }
                }
                if (n->right_child != HT_INVALID_IDX && top < 63) {
                    node_stack[top] = n->right_child;
                    depth_stack[top] = (uint16_t)(d + 1);
                    top++;
                }
                if (n->left_child != HT_INVALID_IDX && top < 63) {
                    node_stack[top] = n->left_child;
                    depth_stack[top] = (uint16_t)(d + 1);
                    top++;
                }
            }
        }
    } else {
        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            if (i == except_idx) {
                continue;
            }
            const ht_node_t *node = &tree->memory.node_pool[i];
            if ((node->flags & HT_NODE_IN_USE) &&
                (node->flags & HT_NODE_IS_LEAF) &&
                (node->stats_idx != HT_INVALID_IDX)) {
                ht_float_t p = ht_tree_node_promise(node, tree->params.n_classes);
                if (p < min_promise) {
                    min_promise = p;
                    min_idx = i;
                }
            }
        }
    }

    if (min_idx == HT_INVALID_IDX) {
        return HT_ERR_OUT_OF_STATS;
    }

    return ht_tree_deactivate_leaf(tree, min_idx);
}

static ht_status_t ht_tree_attempt_split_reg(ht_tree_t *tree, uint16_t leaf_idx, uint16_t depth) {
    ht_node_t *node = &tree->memory.node_pool[leaf_idx];
    if (node->stats_idx == HT_INVALID_IDX) {
        return HT_OK;
    }

    /* Need at least 2 samples and positive variance to split */
    if (node->reg_stats.count <= (ht_float_t)2.0 || node->reg_stats.m2 <= (ht_float_t)0.0) {
        return HT_OK;
    }

    const ht_leaf_stats_t *stats = &tree->memory.stats_pool[node->stats_idx];
    ht_split_candidate_t best, second_best;

    ht_find_best_reg_split_candidates(
        stats,
        &node->reg_stats,
        tree->params.feature_types,
        tree->params.n_features,
        tree->params.min_samples_split,
        tree->params.min_branch_fraction,
        tree->params.merit_preprune,
        &best,
        &second_best
    );

    /* Hoeffding bound with range R = 1.0 */
    ht_float_t eps = ht_hoeffding_bound((ht_float_t)1.0, tree->params.delta, node->reg_stats.count);

    bool should_split = ht_should_split_reg(&best, &second_best, eps, tree->params.tau);

    if (should_split) {
        if (best.feature_idx == HT_NO_FEATURE) {
            return HT_OK;
        }

        if (tree->memory.free_nodes_top < 2) {
            return HT_ERR_OUT_OF_NODES;
        }

        /* Save mother leaf's linear model weights and fmse before recycling */
        ht_float_t parent_weights[HT_MAX_FEATURES];
        memcpy(parent_weights, stats->reg_weights, sizeof(parent_weights));
        ht_float_t parent_intercept = stats->reg_intercept;
        ht_float_t parent_fmse_mean = stats->fmse_mean;
        ht_float_t parent_fmse_model = stats->fmse_model;

        /* Immediate recycling of mother leaf's statistics block */
        uint16_t old_stats = node->stats_idx;
        node->stats_idx = HT_INVALID_IDX;
        ht_stats_free(&tree->memory, old_stats);

        /* Allocate left and right children */
        uint16_t left_idx = ht_node_alloc(&tree->memory);
        uint16_t right_idx = ht_node_alloc(&tree->memory);

        if (left_idx == HT_INVALID_IDX || right_idx == HT_INVALID_IDX) {
            if (left_idx != HT_INVALID_IDX) ht_node_free(&tree->memory, left_idx);
            if (right_idx != HT_INVALID_IDX) ht_node_free(&tree->memory, right_idx);
            return HT_ERR_OUT_OF_NODES;
        }

        /* Convert mother node from leaf to internal branch */
        node->flags &= ~HT_NODE_IS_LEAF;
        if (best.is_categorical) {
            node->flags |= HT_NODE_SPLIT_CAT;
            node->split_val.cat_val = best.split_val.cat_val;
        } else {
            node->flags &= ~HT_NODE_SPLIT_CAT;
            node->split_val.threshold = best.split_val.threshold;
        }
        node->feature_idx = best.feature_idx;
        node->left_child = left_idx;
        node->right_child = right_idx;

        tree->n_decision_nodes++;
        if (tree->n_active_leaves > 0) {
            tree->n_active_leaves--;
        }

        /* Initialize left child */
        ht_node_t *left = &tree->memory.node_pool[left_idx];
        left->reg_stats = best.lhs_var;
        left->flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
        left->left_child = HT_INVALID_IDX;
        left->right_child = HT_INVALID_IDX;
        left->stats_idx = HT_INVALID_IDX;

        /* Initialize right child */
        ht_node_t *right = &tree->memory.node_pool[right_idx];
        right->reg_stats = best.rhs_var;
        right->flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
        right->left_child = HT_INVALID_IDX;
        right->right_child = HT_INVALID_IDX;
        right->stats_idx = HT_INVALID_IDX;

        /* Allocate stats for left child */
        uint16_t l_stats = ht_stats_alloc(&tree->memory);
        if (l_stats == HT_INVALID_IDX) {
            if (ht_tree_deactivate_least_promising_leaf(tree) == HT_OK) {
                l_stats = ht_stats_alloc(&tree->memory);
            }
        }
        if (l_stats != HT_INVALID_IDX) {
            left->stats_idx = l_stats;
            ht_leaf_stats_t *ls = &tree->memory.stats_pool[l_stats];
            ls->last_split_attempt_at = left->reg_stats.count;
            ls->fmse_mean = parent_fmse_mean;
            ls->fmse_model = parent_fmse_model;
            ls->reg_intercept = parent_intercept;
            memcpy(ls->reg_weights, parent_weights, sizeof(parent_weights));
            tree->n_active_leaves++;
        } else {
            tree->n_inactive_leaves++;
        }

        /* Allocate stats for right child */
        uint16_t r_stats = ht_stats_alloc(&tree->memory);
        if (r_stats == HT_INVALID_IDX) {
            if (ht_tree_deactivate_least_promising_leaf_except(tree, left_idx) == HT_OK) {
                r_stats = ht_stats_alloc(&tree->memory);
            }
        }
        if (r_stats != HT_INVALID_IDX) {
            right->stats_idx = r_stats;
            ht_leaf_stats_t *rs = &tree->memory.stats_pool[r_stats];
            rs->last_split_attempt_at = right->reg_stats.count;
            rs->fmse_mean = parent_fmse_mean;
            rs->fmse_model = parent_fmse_model;
            rs->reg_intercept = parent_intercept;
            memcpy(rs->reg_weights, parent_weights, sizeof(parent_weights));
            tree->n_active_leaves++;
        } else {
            tree->n_inactive_leaves++;
        }

        /* HAT: initialize drift tracking for new branch and child nodes */
        if (tree->params.enable_hat && tree->adwin_pool != NULL) {
            tree->node_drift[leaf_idx].alt_tree_idx = HT_INVALID_IDX;
            tree->node_drift[leaf_idx].mean_error = (ht_float_t)0.0;
            tree->node_drift[leaf_idx].error_m2 = (ht_float_t)0.0;
            tree->node_drift[leaf_idx].error_count = 0U;
            if (tree->node_drift[leaf_idx].adwin_idx != HT_INVALID_IDX) {
                ht_adwin_reset(ht_adwin_pool_get(tree->adwin_pool, tree->node_drift[leaf_idx].adwin_idx));
            } else {
                tree->node_drift[leaf_idx].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
            }

            tree->node_drift[left_idx].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
            tree->node_drift[left_idx].alt_tree_idx = HT_INVALID_IDX;
            tree->node_drift[left_idx].mean_error = (ht_float_t)0.0;
            tree->node_drift[left_idx].error_m2 = (ht_float_t)0.0;
            tree->node_drift[left_idx].error_count = 0U;

            tree->node_drift[right_idx].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
            tree->node_drift[right_idx].alt_tree_idx = HT_INVALID_IDX;
            tree->node_drift[right_idx].mean_error = (ht_float_t)0.0;
            tree->node_drift[right_idx].error_m2 = (ht_float_t)0.0;
            tree->node_drift[right_idx].error_count = 0U;
        }

        if ((uint16_t)(depth + 1) > tree->max_depth_seen) {
            tree->max_depth_seen = (uint16_t)(depth + 1);
        }
    }
    return HT_OK;
}

static ht_status_t ht_tree_attempt_split(ht_tree_t *tree, uint16_t leaf_idx, uint16_t depth) {
    ht_node_t *node = &tree->memory.node_pool[leaf_idx];
    if (node->stats_idx == HT_INVALID_IDX) {
        return HT_OK;
    }
    ht_leaf_stats_t *stats = &tree->memory.stats_pool[node->stats_idx];

    /* 1. Check if class distribution is pure */
    uint8_t active_classes = 0;
    ht_float_t total_weight = (ht_float_t)0.0;
    ht_float_t max_class_weight = (ht_float_t)0.0;

    for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
        ht_float_t w = node->class_counts[c];
        total_weight += w;
        if (w > (ht_float_t)0.0) {
            active_classes++;
        }
        if (w > max_class_weight) {
            max_class_weight = w;
        }
    }

    if (active_classes < 2 || total_weight <= (ht_float_t)0.0) {
        return HT_OK;
    }

    /* 2. Check if majority class dominates (max_share_to_split) */
    if (max_class_weight / total_weight > tree->params.max_share_to_split) {
        return HT_OK;
    }

    /* 3. Find top 2 split candidates */
    ht_split_candidate_t best;
    ht_split_candidate_t second_best;

    ht_find_best_split_candidates_ext(stats,
                                      node->class_counts,
                                      tree->params.feature_types,
                                      tree->params.n_features,
                                      tree->params.n_classes,
                                      tree->criterion_vtable,
                                      tree->params.min_branch_fraction,
                                      tree->params.merit_preprune,
                                      tree->params.numeric_obs_type,
                                      &best,
                                      &second_best);

    /* 4. Evaluate Hoeffding Bound */
    ht_float_t r = tree->criterion_vtable->range_of_merit(node->class_counts, tree->params.n_classes);
    ht_float_t eps = ht_hoeffding_bound(r, tree->params.delta, total_weight);

    /* 5. Check split decision */
    bool should_split = ht_should_split(&best, &second_best, eps, tree->params.tau);

    if (should_split) {
        if (best.feature_idx == HT_NO_FEATURE) {
            /* Pre-pruning: null split won by Hoeffding bound -> no split */
            return HT_OK;
        }

        /* Need 2 free nodes in the pool */
        if (tree->memory.free_nodes_top < 2) {
            return HT_ERR_OUT_OF_NODES;
        }

        /* Immediate recycling of mother leaf's statistics block */
        uint16_t old_stats = node->stats_idx;
        node->stats_idx = HT_INVALID_IDX;
        ht_stats_free(&tree->memory, old_stats);

        /* Allocate left and right children */
        uint16_t left_idx = ht_node_alloc(&tree->memory);
        uint16_t right_idx = ht_node_alloc(&tree->memory);

        if (left_idx == HT_INVALID_IDX || right_idx == HT_INVALID_IDX) {
            /* Rollback if allocation fails */
            if (left_idx != HT_INVALID_IDX) ht_node_free(&tree->memory, left_idx);
            if (right_idx != HT_INVALID_IDX) ht_node_free(&tree->memory, right_idx);
            return HT_ERR_OUT_OF_NODES;
        }

        /* Convert mother node from leaf to internal branch */
        node->flags &= ~HT_NODE_IS_LEAF;
        if (best.is_categorical) {
            node->flags |= HT_NODE_SPLIT_CAT;
            node->split_val.cat_val = best.split_val.cat_val;
        } else {
            node->flags &= ~HT_NODE_SPLIT_CAT;
            node->split_val.threshold = best.split_val.threshold;
        }
        node->feature_idx = best.feature_idx;
        node->left_child = left_idx;
        node->right_child = right_idx;

        tree->n_decision_nodes++;
        if (tree->n_active_leaves > 0) {
            tree->n_active_leaves--;
        }

        /* Initialize left child */
        ht_node_t *left = &tree->memory.node_pool[left_idx];
        ht_float_t left_w = (ht_float_t)0.0;
        for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
            left->class_counts[c] = best.lhs_dist[c];
            left_w += best.lhs_dist[c];
        }
        left->flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
        left->left_child = HT_INVALID_IDX;
        left->right_child = HT_INVALID_IDX;
        left->stats_idx = HT_INVALID_IDX;

        /* Initialize right child */
        ht_node_t *right = &tree->memory.node_pool[right_idx];
        ht_float_t right_w = (ht_float_t)0.0;
        for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
            right->class_counts[c] = best.rhs_dist[c];
            right_w += best.rhs_dist[c];
        }
        right->flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
        right->left_child = HT_INVALID_IDX;
        right->right_child = HT_INVALID_IDX;
        right->stats_idx = HT_INVALID_IDX;

        /* Allocate stats for left child */
        uint16_t l_stats = ht_stats_alloc(&tree->memory);
        if (l_stats == HT_INVALID_IDX) {
            if (ht_tree_deactivate_least_promising_leaf(tree) == HT_OK) {
                l_stats = ht_stats_alloc(&tree->memory);
            }
        }
        if (l_stats != HT_INVALID_IDX) {
            left->stats_idx = l_stats;
            tree->memory.stats_pool[l_stats].last_split_attempt_at = left_w;
            tree->n_active_leaves++;
        } else {
            tree->n_inactive_leaves++;
        }

        /* Allocate stats for right child (protecting left child from being immediately deactivated) */
        uint16_t r_stats = ht_stats_alloc(&tree->memory);
        if (r_stats == HT_INVALID_IDX) {
            if (ht_tree_deactivate_least_promising_leaf_except(tree, left_idx) == HT_OK) {
                r_stats = ht_stats_alloc(&tree->memory);
            }
        }
        if (r_stats != HT_INVALID_IDX) {
            right->stats_idx = r_stats;
            tree->memory.stats_pool[r_stats].last_split_attempt_at = right_w;
            tree->n_active_leaves++;
        } else {
            tree->n_inactive_leaves++;
        }

        /* HAT: initialize drift tracking for new branch and child nodes */
        if (tree->params.enable_hat && tree->adwin_pool != NULL) {
            tree->node_drift[leaf_idx].alt_tree_idx = HT_INVALID_IDX;
            tree->node_drift[leaf_idx].mean_error = (ht_float_t)0.0;
            tree->node_drift[leaf_idx].error_count = 0U;
            if (tree->node_drift[leaf_idx].adwin_idx != HT_INVALID_IDX) {
                ht_adwin_reset(ht_adwin_pool_get(tree->adwin_pool, tree->node_drift[leaf_idx].adwin_idx));
            } else {
                tree->node_drift[leaf_idx].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
            }

            tree->node_drift[left_idx].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
            tree->node_drift[left_idx].alt_tree_idx = HT_INVALID_IDX;
            tree->node_drift[left_idx].mean_error = (ht_float_t)0.0;
            tree->node_drift[left_idx].error_count = 0U;

            tree->node_drift[right_idx].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
            tree->node_drift[right_idx].alt_tree_idx = HT_INVALID_IDX;
            tree->node_drift[right_idx].mean_error = (ht_float_t)0.0;
            tree->node_drift[right_idx].error_count = 0U;
        }

        if (depth + 1 > tree->max_depth_seen) {
            tree->max_depth_seen = depth + 1;
        }
    }

    return HT_OK;
}

/* ========================================================================= */
/* --- Public API Implementation ------------------------------------------- */
/* ========================================================================= */

void ht_default_params(ht_config_params_t *params, uint8_t n_features, uint8_t n_classes) {
    if (params == NULL) {
        return;
    }

    memset(params, 0, sizeof(*params));
    params->task_type = HT_TASK_CLASSIFICATION;
    params->n_features = n_features;
    params->n_classes = n_classes;
    for (uint8_t i = 0; i < HT_MAX_FEATURES; ++i) {
        params->feature_types[i] = HT_FEAT_NUMERIC;
    }
    params->criterion = HT_CRIT_INFO_GAIN;
    params->leaf_prediction = HT_LEAF_NBA;
    params->grace_period = 200;
    params->max_depth = 64;
    params->delta = (ht_float_t)1e-7;
    params->tau = (ht_float_t)0.05;
    params->min_branch_fraction = (ht_float_t)0.01;
    params->max_share_to_split = (ht_float_t)0.99;
    params->nb_threshold = (ht_float_t)0.0;
    params->merit_preprune = true;
    params->enable_hat = false;
    params->bootstrap_sampling = false;
    params->drift_window_threshold = 300U;
    params->switch_significance = (ht_float_t)0.05;
    params->adwin_delta = (ht_float_t)0.002;
    params->learning_rate = (ht_float_t)0.01;
    params->l2_regularization = (ht_float_t)1e-4;
    params->model_selector_decay = (ht_float_t)0.95;
    params->min_samples_split = (ht_float_t)5.0;
    params->numeric_obs_type = HT_OBS_GAUSSIAN;
    params->leaf_sgd_normalize = false;
}

void ht_default_hat_params(ht_config_params_t *params, uint8_t n_features, uint8_t n_classes) {
    ht_default_params(params, n_features, n_classes);
    params->enable_hat = true;
    params->bootstrap_sampling = true;
    params->leaf_prediction = HT_LEAF_NBA;
    params->drift_window_threshold = 300U;
    params->switch_significance = (ht_float_t)0.05;
    params->adwin_delta = (ht_float_t)0.002;
}

void ht_default_reg_params(ht_config_params_t *params, uint8_t n_features) {
    if (params == NULL) {
        return;
    }
    memset(params, 0, sizeof(*params));
    params->task_type = HT_TASK_REGRESSION;
    params->n_features = (n_features > HT_MAX_FEATURES) ? (uint8_t)HT_MAX_FEATURES : n_features;
    params->n_classes = 1U;
    for (uint8_t i = 0; i < HT_MAX_FEATURES; ++i) {
        params->feature_types[i] = HT_FEAT_NUMERIC;
    }
    params->criterion = HT_CRIT_VARIANCE_REDUCTION;
    params->leaf_prediction = HT_LEAF_ADAPTIVE;
    params->grace_period = 200U;
    params->max_depth = 64U;
    params->delta = (ht_float_t)1e-7;
    params->tau = (ht_float_t)0.05;
    params->min_branch_fraction = (ht_float_t)0.01;
    params->max_share_to_split = (ht_float_t)0.99;
    params->nb_threshold = (ht_float_t)0.0;
    params->merit_preprune = true;
    params->enable_hat = false;
    params->bootstrap_sampling = false;
    params->drift_window_threshold = 300U;
    params->switch_significance = (ht_float_t)0.05;
    params->adwin_delta = (ht_float_t)0.002;
    params->learning_rate = (ht_float_t)0.01;
    params->l2_regularization = (ht_float_t)1e-4;
    params->model_selector_decay = (ht_float_t)0.95;
    params->min_samples_split = (ht_float_t)5.0;
    params->numeric_obs_type = HT_OBS_GAUSSIAN;
    params->leaf_sgd_normalize = false;
}

void ht_default_hat_reg_params(ht_config_params_t *params, uint8_t n_features) {
    ht_default_reg_params(params, n_features);
    params->enable_hat = true;
    params->bootstrap_sampling = true;
}

ht_status_t ht_tree_init(ht_tree_t *tree, const ht_config_params_t *params) {
    if (tree == NULL || params == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    if (params->n_features == 0 || params->n_features > HT_MAX_FEATURES) {
        return HT_ERR_INVALID_ARG;
    }
    if (params->task_type == HT_TASK_CLASSIFICATION) {
        if (params->n_classes < 2 || params->n_classes > HT_MAX_CLASSES) {
            return HT_ERR_INVALID_ARG;
        }
    }
    if (params->delta <= (ht_float_t)0.0 || params->delta >= (ht_float_t)1.0) {
        return HT_ERR_INVALID_ARG;
    }
    if (params->tau < (ht_float_t)0.0) {
        return HT_ERR_INVALID_ARG;
    }
    if (params->min_branch_fraction < (ht_float_t)0.0 || params->min_branch_fraction > (ht_float_t)0.5) {
        return HT_ERR_INVALID_ARG;
    }
    if (params->max_share_to_split <= (ht_float_t)0.0 || params->max_share_to_split > (ht_float_t)1.0) {
        return HT_ERR_INVALID_ARG;
    }

    const ht_criterion_vtable_t *vtable = ht_criterion_get_vtable(params->criterion);
    if (vtable == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    ht_adwin_pool_t *saved_adwin_pool = tree->adwin_pool;
    memset(tree, 0, sizeof(*tree));
    tree->params = *params;
    tree->criterion_vtable = vtable;
    tree->adwin_pool = saved_adwin_pool;
    tree->rng_state = 123456789U;

    for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
        tree->node_drift[i].adwin_idx = HT_INVALID_IDX;
        tree->node_drift[i].alt_tree_idx = HT_INVALID_IDX;
        tree->node_drift[i].mean_error = (ht_float_t)0.0;
        tree->node_drift[i].error_m2 = (ht_float_t)0.0;
        tree->node_drift[i].error_count = 0U;
    }

    ht_memory_init(&tree->memory);

    /* Allocate root node */
    uint16_t root = ht_node_alloc(&tree->memory);
    if (root == HT_INVALID_IDX) {
        return HT_ERR_OUT_OF_NODES;
    }

    /* Allocate stats for active root leaf */
    uint16_t stats = ht_stats_alloc(&tree->memory);
    if (stats == HT_INVALID_IDX) {
        ht_node_free(&tree->memory, root);
        return HT_ERR_OUT_OF_STATS;
    }

    ht_node_t *root_node = &tree->memory.node_pool[root];
    root_node->stats_idx = stats;
    root_node->flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
    if (params->task_type == HT_TASK_REGRESSION) {
        ht_var_init(&root_node->reg_stats);
    }

    tree->root_idx = root;
    tree->n_active_leaves = 1;
    tree->n_inactive_leaves = 0;
    tree->n_decision_nodes = 0;
    tree->train_weight_seen = (ht_float_t)0.0;
    tree->max_depth_seen = 0;
    tree->n_alternate_trees = 0;
    tree->n_pruned_alternate_trees = 0;
    tree->n_switch_alternate_trees = 0;

    if (tree->params.enable_hat && tree->adwin_pool != NULL) {
        ht_adwin_config_t ad_cfg;
        ht_adwin_default_config(&ad_cfg);
        ad_cfg.delta = tree->params.adwin_delta;
        tree->node_drift[root].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, &ad_cfg);
    }

    return HT_OK;
}

ht_status_t ht_tree_init_hat(ht_tree_t *tree,
                             const ht_config_params_t *params,
                             ht_adwin_pool_t *adwin_pool) {
    if (tree == NULL || params == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    ht_config_params_t hat_params = *params;
    hat_params.enable_hat = true;
    tree->adwin_pool = adwin_pool;
    return ht_tree_init(tree, &hat_params);
}

ht_status_t ht_tree_set_adwin_pool(ht_tree_t *tree, ht_adwin_pool_t *adwin_pool) {
    if (tree == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    tree->adwin_pool = adwin_pool;
    if (adwin_pool != NULL) {
        tree->params.enable_hat = true;
        if (tree->root_idx != HT_INVALID_IDX &&
            tree->node_drift[tree->root_idx].adwin_idx == HT_INVALID_IDX) {
            ht_adwin_config_t ad_cfg;
            ht_adwin_default_config(&ad_cfg);
            ad_cfg.delta = tree->params.adwin_delta;
            tree->node_drift[tree->root_idx].adwin_idx = ht_adwin_pool_alloc(adwin_pool, &ad_cfg);
        }
    }
    return HT_OK;
}

ht_status_t ht_tree_set_prep_pipeline(ht_tree_t *tree, struct ht_prep_pipeline *prep) {
    if (tree == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    tree->prep_pipeline = prep;
    return HT_OK;
}

ht_status_t ht_tree_free_subtree(ht_tree_t *tree, uint16_t subtree_root_idx) {
    if (tree == NULL || subtree_root_idx >= HT_MAX_NODES) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t stack[HT_MAX_NODES];
    uint16_t top = 0;
    stack[top++] = subtree_root_idx;

    while (top > 0) {
        uint16_t curr = stack[--top];
        if (curr >= HT_MAX_NODES) {
            continue;
        }
        ht_node_t *node = &tree->memory.node_pool[curr];
        if (!(node->flags & HT_NODE_IN_USE)) {
            continue;
        }

        /* 1. Free alternate tree if one exists */
        if (tree->node_drift[curr].alt_tree_idx != HT_INVALID_IDX) {
            uint16_t alt_idx = tree->node_drift[curr].alt_tree_idx;
            tree->node_drift[curr].alt_tree_idx = HT_INVALID_IDX;
            stack[top++] = alt_idx;
        }

        /* 2. Free children if branch */
        if (!(node->flags & HT_NODE_IS_LEAF)) {
            if (node->left_child != HT_INVALID_IDX) {
                stack[top++] = node->left_child;
                node->left_child = HT_INVALID_IDX;
            }
            if (node->right_child != HT_INVALID_IDX) {
                stack[top++] = node->right_child;
                node->right_child = HT_INVALID_IDX;
            }
            if (tree->n_decision_nodes > 0) {
                tree->n_decision_nodes--;
            }
        }

        /* 3. Free stats if allocated */
        if (node->stats_idx != HT_INVALID_IDX) {
            ht_stats_free(&tree->memory, node->stats_idx);
            node->stats_idx = HT_INVALID_IDX;
            if (node->flags & HT_NODE_IS_LEAF) {
                if (tree->n_active_leaves > 0) {
                    tree->n_active_leaves--;
                }
            }
        } else if (node->flags & HT_NODE_IS_LEAF) {
            if (tree->n_inactive_leaves > 0) {
                tree->n_inactive_leaves--;
            }
        }

        /* 4. Free ADWIN detector */
        if (tree->node_drift[curr].adwin_idx != HT_INVALID_IDX) {
            if (tree->adwin_pool != NULL) {
                ht_adwin_pool_free(tree->adwin_pool, tree->node_drift[curr].adwin_idx);
            }
            tree->node_drift[curr].adwin_idx = HT_INVALID_IDX;
        }
        tree->node_drift[curr].mean_error = (ht_float_t)0.0;
        tree->node_drift[curr].error_count = 0U;

        /* 5. Free structural node */
        ht_node_free(&tree->memory, curr);
    }

    return HT_OK;
}

uint16_t ht_tree_route(const ht_tree_t *tree,
                       const ht_value_t x[HT_MAX_FEATURES],
                       uint16_t *out_parent_idx,
                       uint8_t *out_parent_branch,
                       uint16_t *out_depth) {
    if (tree == NULL || x == NULL || tree->root_idx == HT_INVALID_IDX) {
        return HT_INVALID_IDX;
    }

    uint16_t curr_idx = tree->root_idx;
    uint16_t parent_idx = HT_INVALID_IDX;
    uint8_t parent_branch = 0;
    uint16_t depth = 0;

    while (curr_idx != HT_INVALID_IDX) {
        const ht_node_t *node = &tree->memory.node_pool[curr_idx];
        if (node->flags & HT_NODE_IS_LEAF) {
            if (out_parent_idx != NULL) {
                *out_parent_idx = parent_idx;
            }
            if (out_parent_branch != NULL) {
                *out_parent_branch = parent_branch;
            }
            if (out_depth != NULL) {
                *out_depth = depth;
            }
            return curr_idx;
        }

        uint8_t f = node->feature_idx;
        uint16_t next_idx;
        uint8_t branch = 0;

        if (node->flags & HT_NODE_SPLIT_CAT) {
            int32_t cat = x[f].cat;
            if (cat < 0 || cat >= (int32_t)HT_MAX_CATEGORIES) {
                /* Missing value fallback: most common child */
                next_idx = ht_most_common_child(tree, node);
                branch = (next_idx == node->left_child) ? 0 : 1;
            } else if (cat == node->split_val.cat_val) {
                next_idx = node->left_child;
                branch = 0;
            } else {
                next_idx = node->right_child;
                branch = 1;
            }
        } else {
            ht_float_t num = x[f].num;
            if (HT_ISNAN(num)) {
                /* Missing value fallback: most common child */
                next_idx = ht_most_common_child(tree, node);
                branch = (next_idx == node->left_child) ? 0 : 1;
            } else if (num <= node->split_val.threshold) {
                next_idx = node->left_child;
                branch = 0;
            } else {
                next_idx = node->right_child;
                branch = 1;
            }
        }

        parent_idx = curr_idx;
        parent_branch = branch;
        curr_idx = next_idx;
        depth++;
    }

    return HT_INVALID_IDX;
}

static uint32_t ht_prng_next(uint32_t *state) {
    if (*state == 0U) {
        *state = 123456789U;
    }
    *state = (*state * 1664525U) + 1013904223U;
    return *state;
}

static uint32_t ht_poisson_1(uint32_t *rng_state) {
    const ht_float_t exp_minus_1 = (ht_float_t)0.36787944117144232;
    ht_float_t p = (ht_float_t)1.0;
    uint32_t k = 0;
    do {
        k++;
        uint32_t r = ht_prng_next(rng_state);
        ht_float_t u = ((ht_float_t)(r >> 8) + (ht_float_t)0.5) / (ht_float_t)16777216.0;
        p *= u;
    } while (p > exp_minus_1 && k < 100);
    return k - 1;
}

static uint16_t ht_tree_route_from(const ht_tree_t *tree,
                                   uint16_t start_idx,
                                   const ht_value_t x[HT_MAX_FEATURES]) {
    uint16_t curr_idx = start_idx;
    while (curr_idx != HT_INVALID_IDX) {
        const ht_node_t *node = &tree->memory.node_pool[curr_idx];
        if (node->flags & HT_NODE_IS_LEAF) {
            return curr_idx;
        }

        uint8_t f = node->feature_idx;
        uint16_t next_idx;
        if (node->flags & HT_NODE_SPLIT_CAT) {
            int32_t cat = x[f].cat;
            if (cat < 0 || cat >= (int32_t)HT_MAX_CATEGORIES) {
                next_idx = ht_most_common_child(tree, node);
            } else if (cat == node->split_val.cat_val) {
                next_idx = node->left_child;
            } else {
                next_idx = node->right_child;
            }
        } else {
            ht_float_t num = x[f].num;
            if (HT_ISNAN(num)) {
                next_idx = ht_most_common_child(tree, node);
            } else if (num <= node->split_val.threshold) {
                next_idx = node->left_child;
            } else {
                next_idx = node->right_child;
            }
        }
        curr_idx = next_idx;
    }
    return HT_INVALID_IDX;
}

ht_status_t ht_tree_predict_proba(const ht_tree_t *tree,
                                  const ht_value_t x[HT_MAX_FEATURES],
                                  ht_float_t proba_out[HT_MAX_CLASSES]) {
    if (tree == NULL || x == NULL || proba_out == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    if (tree->root_idx == HT_INVALID_IDX) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            proba_out[c] = (ht_float_t)0.0;
        }
        return HT_ERR_INVALID_ARG;
    }

    const ht_value_t *x_in = x;
    ht_value_t x_prep[HT_MAX_FEATURES];
    if (tree->prep_pipeline != NULL) {
        ht_prep_transform_one(tree->prep_pipeline, x, x_prep);
        x_in = x_prep;
    }

    if (!tree->params.enable_hat || tree->adwin_pool == NULL) {
        uint16_t leaf_idx = ht_tree_route(tree, x_in, NULL, NULL, NULL);
        if (leaf_idx == HT_INVALID_IDX) {
            for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
                proba_out[c] = (ht_float_t)0.0;
            }
            return HT_ERR_INVALID_ARG;
        }
        ht_predict_proba_at_leaf(tree, leaf_idx, x_in, proba_out);
        return HT_OK;
    }

    /* HAT routing: traverse main tree and any active alternate subtrees */
    uint16_t root_queue[64];
    uint8_t q_head = 0, q_tail = 0;
    root_queue[q_tail++] = tree->root_idx;

    uint16_t reached_leaves[64];
    uint8_t n_reached = 0;

    while (q_head < q_tail && n_reached < 64) {
        uint16_t curr = root_queue[q_head++];
        while (curr != HT_INVALID_IDX) {
            const ht_node_t *node = &tree->memory.node_pool[curr];
            if (node->flags & HT_NODE_IS_LEAF) {
                reached_leaves[n_reached++] = curr;
                break;
            }

            if (tree->node_drift[curr].alt_tree_idx != HT_INVALID_IDX && q_tail < 64) {
                root_queue[q_tail++] = tree->node_drift[curr].alt_tree_idx;
            }

            uint8_t f = node->feature_idx;
            uint16_t next_idx;
            if (node->flags & HT_NODE_SPLIT_CAT) {
                int32_t cat = x_in[f].cat;
                if (cat < 0 || cat >= (int32_t)HT_MAX_CATEGORIES) {
                    next_idx = ht_most_common_child(tree, node);
                } else if (cat == node->split_val.cat_val) {
                    next_idx = node->left_child;
                } else {
                    next_idx = node->right_child;
                }
            } else {
                ht_float_t num = x_in[f].num;
                if (HT_ISNAN(num)) {
                    next_idx = ht_most_common_child(tree, node);
                } else if (num <= node->split_val.threshold) {
                    next_idx = node->left_child;
                } else {
                    next_idx = node->right_child;
                }
            }
            curr = next_idx;
        }
    }

    if (n_reached == 0) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            proba_out[c] = (ht_float_t)0.0;
        }
        return HT_ERR_INVALID_ARG;
    }

    if (n_reached == 1) {
        ht_predict_proba_at_leaf(tree, reached_leaves[0], x_in, proba_out);
        return HT_OK;
    }

    ht_float_t leaf_probas[64 * HT_MAX_CLASSES];
    ht_float_t leaf_errors[64];
    for (uint8_t i = 0; i < n_reached; ++i) {
        uint16_t l_idx = reached_leaves[i];
        ht_predict_proba_at_leaf(tree, l_idx, x_in, &leaf_probas[(size_t)i * HT_MAX_CLASSES]);
        leaf_errors[i] = tree->node_drift[l_idx].mean_error;
    }

    ht_hat_combine_leaf_predictions(leaf_probas, leaf_errors, n_reached, tree->params.n_classes, proba_out);
    return HT_OK;
}

uint8_t ht_tree_predict(const ht_tree_t *tree,
                        const ht_value_t x[HT_MAX_FEATURES]) {
    ht_float_t probas[HT_MAX_CLASSES];
    ht_status_t status = ht_tree_predict_proba(tree, x, probas);
    if (status != HT_OK || tree == NULL) {
        return 0;
    }

    uint8_t best_c = 0;
    ht_float_t best_p = (ht_float_t)-1.0;
    for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
        if (probas[c] > best_p) {
            best_p = probas[c];
            best_c = c;
        }
    }

    return best_c;
}

static ht_status_t ht_tree_learn_one_subtree(ht_tree_t *tree,
                                             uint16_t sub_root,
                                             const ht_value_t x[HT_MAX_FEATURES],
                                             uint8_t y,
                                             ht_float_t weight,
                                             uint16_t depth) {
    if (sub_root == HT_INVALID_IDX) {
        return HT_OK;
    }

    uint8_t detec_in = 0;
    if (tree->params.enable_hat && tree->adwin_pool != NULL) {
        uint16_t pred_leaf = ht_tree_route_from(tree, sub_root, x);
        if (pred_leaf != HT_INVALID_IDX) {
            ht_float_t probas[HT_MAX_CLASSES];
            ht_predict_proba_at_leaf(tree, pred_leaf, x, probas);
            uint8_t y_pred = 0;
            ht_float_t best_p = (ht_float_t)-1.0;
            for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
                if (probas[c] > best_p) {
                    best_p = probas[c];
                    y_pred = c;
                }
            }
            detec_in = (y_pred == y) ? 0U : 1U;
        }
    }

    uint16_t curr_idx = sub_root;
    uint16_t parent_idx = HT_INVALID_IDX;
    uint8_t parent_branch = 0;

    while (curr_idx != HT_INVALID_IDX) {
        ht_node_t *node = &tree->memory.node_pool[curr_idx];
        if (node->flags & HT_NODE_IS_LEAF) {
            break;
        }

        node->class_counts[y] += weight;

        if (tree->params.enable_hat && tree->adwin_pool != NULL) {
            ht_node_drift_t *nd = &tree->node_drift[curr_idx];
            bool error_drift = false;
            if (nd->adwin_idx != HT_INVALID_IDX) {
                ht_adwin_t *ad = ht_adwin_pool_get(tree->adwin_pool, nd->adwin_idx);
                error_drift = ht_hat_check_error_drift(ad, &nd->mean_error, &nd->error_count, detec_in);
            }

            if (error_drift) {
                if (nd->alt_tree_idx != HT_INVALID_IDX) {
                    ht_tree_free_subtree(tree, nd->alt_tree_idx);
                    tree->n_pruned_alternate_trees++;
                    nd->alt_tree_idx = HT_INVALID_IDX;
                }

                uint16_t alt_leaf = ht_node_alloc(&tree->memory);
                if (alt_leaf != HT_INVALID_IDX) {
                    uint16_t alt_stats = ht_stats_alloc(&tree->memory);
                    if (alt_stats == HT_INVALID_IDX) {
                        if (ht_tree_deactivate_least_promising_leaf(tree) == HT_OK) {
                            alt_stats = ht_stats_alloc(&tree->memory);
                        }
                    }
                    if (alt_stats != HT_INVALID_IDX) {
                        ht_node_t *al = &tree->memory.node_pool[alt_leaf];
                        al->stats_idx = alt_stats;
                        tree->memory.stats_pool[alt_stats].last_split_attempt_at = (ht_float_t)0.0;
                        tree->n_active_leaves++;
                    } else {
                        tree->n_inactive_leaves++;
                    }

                    tree->node_drift[alt_leaf].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
                    tree->node_drift[alt_leaf].alt_tree_idx = HT_INVALID_IDX;
                    tree->node_drift[alt_leaf].mean_error = (ht_float_t)0.0;
                    tree->node_drift[alt_leaf].error_count = 0U;

                    nd->alt_tree_idx = alt_leaf;
                    tree->n_alternate_trees++;
                }
            } else if (nd->alt_tree_idx != HT_INVALID_IDX) {
                uint16_t alt_idx = nd->alt_tree_idx;
                ht_node_drift_t *alt_nd = &tree->node_drift[alt_idx];

                ht_hat_action_t action = ht_hat_evaluate_alternate(
                    nd->mean_error,
                    nd->error_count,
                    alt_nd->mean_error,
                    alt_nd->error_count,
                    tree->params.drift_window_threshold,
                    tree->params.switch_significance
                );

                if (action == HT_HAT_ACTION_SWITCH) {
                    if (parent_idx != HT_INVALID_IDX) {
                        ht_node_t *p_node = &tree->memory.node_pool[parent_idx];
                        if (parent_branch == 0) {
                            p_node->left_child = alt_idx;
                        } else {
                            p_node->right_child = alt_idx;
                        }
                    } else if (tree->root_idx == curr_idx) {
                        tree->root_idx = alt_idx;
                    }

                    nd->alt_tree_idx = HT_INVALID_IDX;
                    ht_tree_free_subtree(tree, curr_idx);
                    tree->n_switch_alternate_trees++;

                    curr_idx = alt_idx;
                    node = &tree->memory.node_pool[curr_idx];
                    if (node->flags & HT_NODE_IS_LEAF) {
                        break;
                    }
                } else if (action == HT_HAT_ACTION_PRUNE) {
                    ht_tree_free_subtree(tree, alt_idx);
                    nd->alt_tree_idx = HT_INVALID_IDX;
                    tree->n_pruned_alternate_trees++;
                }
            }

            if (nd->alt_tree_idx != HT_INVALID_IDX) {
                ht_tree_learn_one_subtree(tree, nd->alt_tree_idx, x, y, weight, depth);
            }
        }

        uint8_t f = node->feature_idx;
        uint16_t next_idx;
        uint8_t branch;

        if (node->flags & HT_NODE_SPLIT_CAT) {
            int32_t cat = x[f].cat;
            if (cat < 0 || cat >= (int32_t)HT_MAX_CATEGORIES) {
                next_idx = ht_most_common_child(tree, node);
                branch = (next_idx == node->left_child) ? 0 : 1;
            } else if (cat == node->split_val.cat_val) {
                next_idx = node->left_child;
                branch = 0;
            } else {
                next_idx = node->right_child;
                branch = 1;
            }
        } else {
            ht_float_t num = x[f].num;
            if (HT_ISNAN(num)) {
                next_idx = ht_most_common_child(tree, node);
                branch = (next_idx == node->left_child) ? 0 : 1;
            } else if (num <= node->split_val.threshold) {
                next_idx = node->left_child;
                branch = 0;
            } else {
                next_idx = node->right_child;
                branch = 1;
            }
        }

        parent_idx = curr_idx;
        parent_branch = branch;
        curr_idx = next_idx;
        depth++;
    }

    uint16_t leaf_idx = curr_idx;
    ht_node_t *leaf = &tree->memory.node_pool[leaf_idx];

    if (tree->params.enable_hat && tree->adwin_pool != NULL) {
        ht_node_drift_t *leaf_nd = &tree->node_drift[leaf_idx];
        if (leaf_nd->adwin_idx != HT_INVALID_IDX) {
            ht_adwin_t *ad = ht_adwin_pool_get(tree->adwin_pool, leaf_nd->adwin_idx);
            ht_hat_check_error_drift(ad, &leaf_nd->mean_error, &leaf_nd->error_count, detec_in);
        }
    }

    /* Active leaf learning */
    if (leaf->stats_idx != HT_INVALID_IDX) {
        ht_leaf_stats_t *stats = &tree->memory.stats_pool[leaf->stats_idx];

        /* Prequential performance tracking for NBA before stats update */
        if (tree->params.leaf_prediction == HT_LEAF_NBA) {
            ht_float_t leaf_total_w = (ht_float_t)0.0;
            for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
                leaf_total_w += leaf->class_counts[c];
            }

            if (leaf_total_w <= (ht_float_t)0.0) {
                stats->mc_correct_weight += weight;
            } else {
                uint8_t mc_pred = ht_leaf_predict_mc_class(leaf, tree->params.n_classes);
                if (mc_pred == y) {
                    stats->mc_correct_weight += weight;
                }
            }

            if (leaf_total_w > (ht_float_t)0.0) {
                uint8_t nb_pred = 0;
                if (ht_leaf_predict_nb_class(tree, leaf, stats, x, &nb_pred)) {
                    if (nb_pred == y) {
                        stats->nb_correct_weight += weight;
                    }
                }
            }
        }

        /* Update leaf class distribution */
        leaf->class_counts[y] += weight;

        /* Update attribute observers */
        for (uint8_t f = 0; f < tree->params.n_features; ++f) {
            if (tree->params.feature_types[f] == HT_FEAT_NUMERIC) {
                if (!HT_ISNAN(x[f].num)) {
                    if (tree->params.numeric_obs_type == HT_OBS_HISTOGRAM) {
                        ht_hist_update(&stats->observers[f].hist, y, x[f].num, weight);
                    } else {
                        ht_gaussian_update(stats->observers[f].num, y, x[f].num, weight);
                    }
                }
            } else {
                int32_t cat = x[f].cat;
                if (cat >= 0 && cat < (int32_t)HT_MAX_CATEGORIES) {
                    ht_categorical_update(&stats->observers[f].cat, y, cat, weight);
                }
            }
        }

        /* Check max depth */
        if (depth >= tree->params.max_depth) {
            ht_tree_deactivate_leaf(tree, leaf_idx);
            return HT_OK;
        }

        /* Check grace period for split attempt */
        ht_float_t weight_seen = (ht_float_t)0.0;
        for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
            weight_seen += leaf->class_counts[c];
        }

        ht_float_t weight_diff = weight_seen - stats->last_split_attempt_at;
        if (weight_diff >= (ht_float_t)tree->params.grace_period) {
            stats->last_split_attempt_at = weight_seen;
            ht_tree_attempt_split(tree, leaf_idx, depth);
        }
    } else {
        /* Inactive leaf: update class distribution only */
        leaf->class_counts[y] += weight;
    }

    return HT_OK;
}

ht_status_t ht_tree_learn_one(ht_tree_t *tree,
                              const ht_value_t x[HT_MAX_FEATURES],
                              uint8_t y,
                              ht_float_t weight) {
    if (tree == NULL || x == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    if (y >= tree->params.n_classes) {
        return HT_ERR_INVALID_ARG;
    }
    if (weight <= (ht_float_t)0.0) {
        return HT_OK;
    }
    if (tree->root_idx == HT_INVALID_IDX) {
        return HT_ERR_INVALID_ARG;
    }

    const ht_value_t *x_in = x;
    ht_value_t x_prep[HT_MAX_FEATURES];
    if (tree->prep_pipeline != NULL) {
        ht_prep_learn_transform_one(tree->prep_pipeline, x, x_prep, weight);
        x_in = x_prep;
    }

    if (tree->params.enable_hat && tree->params.bootstrap_sampling) {
        uint32_t k = ht_poisson_1(&tree->rng_state);
        if (k > 0) {
            weight *= (ht_float_t)k;
        }
    }

    tree->train_weight_seen += weight;
    return ht_tree_learn_one_subtree(tree, tree->root_idx, x_in, y, weight, 0);
}

ht_float_t ht_tree_predict_reg(const ht_tree_t *tree, const ht_value_t x[HT_MAX_FEATURES]) {
    if (tree == NULL || x == NULL || tree->root_idx == HT_INVALID_IDX) {
        return (ht_float_t)0.0;
    }

    const ht_value_t *x_in = x;
    ht_value_t x_prep[HT_MAX_FEATURES];
    if (tree->prep_pipeline != NULL) {
        ht_prep_transform_one(tree->prep_pipeline, x, x_prep);
        x_in = x_prep;
    }

    if (tree->params.enable_hat && tree->adwin_pool != NULL) {
        uint16_t curr_idx = tree->root_idx;
        ht_float_t sum_pred = (ht_float_t)0.0;
        uint32_t n_leaves = 0;

        while (curr_idx != HT_INVALID_IDX) {
            const ht_node_t *node = &tree->memory.node_pool[curr_idx];
            if (node->flags & HT_NODE_IS_LEAF) {
                sum_pred += ht_predict_reg_at_leaf(tree, curr_idx, x_in);
                n_leaves++;
                break;
            }

            const ht_node_drift_t *nd = &tree->node_drift[curr_idx];
            if (nd->alt_tree_idx != HT_INVALID_IDX) {
                uint16_t alt_leaf = ht_tree_route_from(tree, nd->alt_tree_idx, x_in);
                if (alt_leaf != HT_INVALID_IDX) {
                    sum_pred += ht_predict_reg_at_leaf(tree, alt_leaf, x_in);
                    n_leaves++;
                }
            }

            uint8_t f = node->feature_idx;
            uint16_t next_idx;
            if (node->flags & HT_NODE_SPLIT_CAT) {
                int32_t cat = x_in[f].cat;
                if (cat < 0 || cat >= (int32_t)HT_MAX_CATEGORIES) {
                    next_idx = ht_most_common_child(tree, node);
                } else if (cat == node->split_val.cat_val) {
                    next_idx = node->left_child;
                } else {
                    next_idx = node->right_child;
                }
            } else {
                ht_float_t num = x_in[f].num;
                if (HT_ISNAN(num)) {
                    next_idx = ht_most_common_child(tree, node);
                } else if (num <= node->split_val.threshold) {
                    next_idx = node->left_child;
                } else {
                    next_idx = node->right_child;
                }
            }
            curr_idx = next_idx;
        }

        if (n_leaves == 0) {
            return (ht_float_t)0.0;
        }
        return sum_pred / (ht_float_t)n_leaves;
    }

    uint16_t leaf_idx = ht_tree_route(tree, x_in, NULL, NULL, NULL);
    return ht_predict_reg_at_leaf(tree, leaf_idx, x_in);
}

static ht_status_t ht_tree_learn_one_reg_subtree(ht_tree_t *tree,
                                                 uint16_t sub_root,
                                                 const ht_value_t x[HT_MAX_FEATURES],
                                                 ht_float_t y,
                                                 ht_float_t weight,
                                                 uint16_t depth) {
    if (sub_root == HT_INVALID_IDX) {
        return HT_OK;
    }

    ht_float_t abs_error = (ht_float_t)0.0;
    if (tree->params.enable_hat && tree->adwin_pool != NULL) {
        uint16_t pred_leaf = ht_tree_route_from(tree, sub_root, x);
        if (pred_leaf != HT_INVALID_IDX) {
            ht_float_t y_pred = ht_predict_reg_at_leaf(tree, pred_leaf, x);
            abs_error = HT_FABS(y - y_pred);
        }
    }

    uint16_t curr_idx = sub_root;
    uint16_t parent_idx = HT_INVALID_IDX;
    uint8_t parent_branch = 0;

    while (curr_idx != HT_INVALID_IDX) {
        ht_node_t *node = &tree->memory.node_pool[curr_idx];
        if (node->flags & HT_NODE_IS_LEAF) {
            break;
        }

        ht_var_update(&node->reg_stats, y, weight);

        if (tree->params.enable_hat && tree->adwin_pool != NULL) {
            ht_node_drift_t *nd = &tree->node_drift[curr_idx];
            bool error_drift = false;
            if (nd->adwin_idx != HT_INVALID_IDX) {
                ht_adwin_t *ad = ht_adwin_pool_get(tree->adwin_pool, nd->adwin_idx);
                error_drift = ht_hat_check_reg_error_drift(ad, &nd->mean_error, &nd->error_m2, &nd->error_count, abs_error);
            }

            if (error_drift) {
                if (nd->alt_tree_idx != HT_INVALID_IDX) {
                    ht_tree_free_subtree(tree, nd->alt_tree_idx);
                    tree->n_pruned_alternate_trees++;
                    nd->alt_tree_idx = HT_INVALID_IDX;
                }

                uint16_t alt_leaf = ht_node_alloc(&tree->memory);
                if (alt_leaf != HT_INVALID_IDX) {
                    uint16_t alt_stats = ht_stats_alloc(&tree->memory);
                    if (alt_stats == HT_INVALID_IDX) {
                        if (ht_tree_deactivate_least_promising_leaf(tree) == HT_OK) {
                            alt_stats = ht_stats_alloc(&tree->memory);
                        }
                    }
                    if (alt_stats != HT_INVALID_IDX) {
                        ht_node_t *al = &tree->memory.node_pool[alt_leaf];
                        al->stats_idx = alt_stats;
                        ht_leaf_stats_t *als = &tree->memory.stats_pool[alt_stats];
                        als->last_split_attempt_at = (ht_float_t)0.0;
                        als->fmse_mean = (ht_float_t)0.0;
                        als->fmse_model = (ht_float_t)0.0;
                        als->reg_intercept = (ht_float_t)0.0;
                        memset(als->reg_weights, 0, sizeof(als->reg_weights));
                        tree->n_active_leaves++;
                    } else {
                        tree->n_inactive_leaves++;
                    }

                    tree->node_drift[alt_leaf].adwin_idx = ht_adwin_pool_alloc(tree->adwin_pool, NULL);
                    tree->node_drift[alt_leaf].alt_tree_idx = HT_INVALID_IDX;
                    tree->node_drift[alt_leaf].mean_error = (ht_float_t)0.0;
                    tree->node_drift[alt_leaf].error_m2 = (ht_float_t)0.0;
                    tree->node_drift[alt_leaf].error_count = 0U;

                    nd->alt_tree_idx = alt_leaf;
                    tree->n_alternate_trees++;
                }
            } else if (nd->alt_tree_idx != HT_INVALID_IDX) {
                uint16_t alt_idx = nd->alt_tree_idx;
                ht_node_drift_t *alt_nd = &tree->node_drift[alt_idx];

                ht_float_t var_old = (nd->error_count > 1) ? (nd->error_m2 / (ht_float_t)(nd->error_count - 1)) : (ht_float_t)0.0;
                ht_float_t var_alt = (alt_nd->error_count > 1) ? (alt_nd->error_m2 / (ht_float_t)(alt_nd->error_count - 1)) : (ht_float_t)0.0;

                ht_hat_action_t action = ht_hat_evaluate_reg_alternate(
                    nd->mean_error,
                    var_old,
                    nd->error_count,
                    alt_nd->mean_error,
                    var_alt,
                    alt_nd->error_count,
                    tree->params.drift_window_threshold,
                    tree->params.switch_significance
                );

                if (action == HT_HAT_ACTION_SWITCH) {
                    if (parent_idx != HT_INVALID_IDX) {
                        ht_node_t *p_node = &tree->memory.node_pool[parent_idx];
                        if (parent_branch == 0) {
                            p_node->left_child = alt_idx;
                        } else {
                            p_node->right_child = alt_idx;
                        }
                    } else if (tree->root_idx == curr_idx) {
                        tree->root_idx = alt_idx;
                    }

                    nd->alt_tree_idx = HT_INVALID_IDX;
                    ht_tree_free_subtree(tree, curr_idx);
                    tree->n_switch_alternate_trees++;

                    curr_idx = alt_idx;
                    node = &tree->memory.node_pool[curr_idx];
                    if (node->flags & HT_NODE_IS_LEAF) {
                        break;
                    }
                } else if (action == HT_HAT_ACTION_PRUNE) {
                    ht_tree_free_subtree(tree, alt_idx);
                    nd->alt_tree_idx = HT_INVALID_IDX;
                    tree->n_pruned_alternate_trees++;
                }
            }

            if (nd->alt_tree_idx != HT_INVALID_IDX) {
                ht_tree_learn_one_reg_subtree(tree, nd->alt_tree_idx, x, y, weight, depth);
            }
        }

        uint8_t f = node->feature_idx;
        uint16_t next_idx;
        uint8_t branch;

        if (node->flags & HT_NODE_SPLIT_CAT) {
            int32_t cat = x[f].cat;
            if (cat < 0 || cat >= (int32_t)HT_MAX_CATEGORIES) {
                next_idx = ht_most_common_child(tree, node);
                branch = (next_idx == node->left_child) ? 0 : 1;
            } else if (cat == node->split_val.cat_val) {
                next_idx = node->left_child;
                branch = 0;
            } else {
                next_idx = node->right_child;
                branch = 1;
            }
        } else {
            ht_float_t num = x[f].num;
            if (HT_ISNAN(num)) {
                next_idx = ht_most_common_child(tree, node);
                branch = (next_idx == node->left_child) ? 0 : 1;
            } else if (num <= node->split_val.threshold) {
                next_idx = node->left_child;
                branch = 0;
            } else {
                next_idx = node->right_child;
                branch = 1;
            }
        }

        parent_idx = curr_idx;
        parent_branch = branch;
        curr_idx = next_idx;
        depth++;
    }

    uint16_t leaf_idx = curr_idx;
    ht_node_t *leaf = &tree->memory.node_pool[leaf_idx];

    if (tree->params.enable_hat && tree->adwin_pool != NULL) {
        ht_node_drift_t *leaf_nd = &tree->node_drift[leaf_idx];
        if (leaf_nd->adwin_idx != HT_INVALID_IDX) {
            ht_adwin_t *ad = ht_adwin_pool_get(tree->adwin_pool, leaf_nd->adwin_idx);
            ht_hat_check_reg_error_drift(ad, &leaf_nd->mean_error, &leaf_nd->error_m2, &leaf_nd->error_count, abs_error);
        }
    }

    if (leaf->stats_idx != HT_INVALID_IDX) {
        ht_leaf_stats_t *stats = &tree->memory.stats_pool[leaf->stats_idx];

        /* 1. Update fMSE if adaptive */
        if (tree->params.leaf_prediction == HT_LEAF_ADAPTIVE) {
            ht_float_t pred_mean = leaf->reg_stats.mean;
            ht_float_t pred_model = ht_leaf_predict_linear_model(tree, stats, x);
            ht_float_t err_mean = y - pred_mean;
            ht_float_t err_model = y - pred_model;
            stats->fmse_mean = tree->params.model_selector_decay * stats->fmse_mean + (err_mean * err_mean);
            stats->fmse_model = tree->params.model_selector_decay * stats->fmse_model + (err_model * err_model);
        }

        /* 2. Update linear model via online SGD if MODEL or ADAPTIVE */
        if (tree->params.leaf_prediction == HT_LEAF_MODEL || tree->params.leaf_prediction == HT_LEAF_ADAPTIVE) {
            ht_float_t pred = ht_leaf_predict_linear_model(tree, stats, x);
            ht_float_t err = y - pred;
            ht_float_t n = leaf->reg_stats.count;
            ht_float_t eta = tree->params.learning_rate / ((ht_float_t)1.0 + tree->params.l2_regularization * n);

            stats->reg_intercept += eta * weight * err;
            if (stats->reg_intercept > (ht_float_t)1e12) stats->reg_intercept = (ht_float_t)1e12;
            else if (stats->reg_intercept < (ht_float_t)-1e12) stats->reg_intercept = (ht_float_t)-1e12;

            ht_float_t l2_decay = (ht_float_t)1.0 - eta * tree->params.l2_regularization;
            if (l2_decay < (ht_float_t)0.0) l2_decay = (ht_float_t)0.0;
            for (uint8_t f = 0; f < tree->params.n_features; ++f) {
                if (tree->params.feature_types[f] == HT_FEAT_NUMERIC) {
                    if (!HT_ISNAN(x[f].num)) {
                        ht_float_t val = x[f].num;
                        if (tree->params.leaf_sgd_normalize) {
                            val = ht_tree_normalize_feature_for_sgd(tree, stats, f, val);
                        }
                        stats->reg_weights[f] = stats->reg_weights[f] * l2_decay + eta * weight * err * val;
                        if (stats->reg_weights[f] > (ht_float_t)1e12) stats->reg_weights[f] = (ht_float_t)1e12;
                        else if (stats->reg_weights[f] < (ht_float_t)-1e12) stats->reg_weights[f] = (ht_float_t)-1e12;
                    }
                }
            }
        }

        /* 3. Update target statistics */
        ht_var_update(&leaf->reg_stats, y, weight);

        /* 4. Update feature observers */
        for (uint8_t f = 0; f < tree->params.n_features; ++f) {
            if (tree->params.feature_types[f] == HT_FEAT_NUMERIC) {
                if (!HT_ISNAN(x[f].num)) {
                    ht_reg_numeric_update(&stats->observers[f].reg_num, x[f].num, y, weight);
                }
            } else {
                int32_t cat = x[f].cat;
                if (cat >= 0 && cat < (int32_t)HT_MAX_CATEGORIES) {
                    ht_reg_categorical_update(&stats->observers[f].reg_cat, cat, y, weight);
                }
            }
        }

        /* 5. Check max depth */
        if (depth >= tree->params.max_depth) {
            ht_tree_deactivate_leaf(tree, leaf_idx);
            return HT_OK;
        }

        /* 6. Check grace period */
        ht_float_t weight_diff = leaf->reg_stats.count - stats->last_split_attempt_at;
        if (weight_diff >= (ht_float_t)tree->params.grace_period) {
            stats->last_split_attempt_at = leaf->reg_stats.count;
            ht_tree_attempt_split_reg(tree, leaf_idx, depth);
        }
    } else {
        ht_var_update(&leaf->reg_stats, y, weight);
    }

    return HT_OK;
}

ht_status_t ht_tree_learn_one_reg(ht_tree_t *tree,
                                  const ht_value_t x[HT_MAX_FEATURES],
                                  ht_float_t y,
                                  ht_float_t weight) {
    if (tree == NULL || x == NULL || weight <= (ht_float_t)0.0 || HT_ISNAN(y)) {
        return HT_ERR_INVALID_ARG;
    }
    if (tree->params.task_type != HT_TASK_REGRESSION) {
        return HT_ERR_INVALID_ARG;
    }

    const ht_value_t *x_in = x;
    ht_value_t x_prep[HT_MAX_FEATURES];
    if (tree->prep_pipeline != NULL) {
        ht_prep_learn_transform_one(tree->prep_pipeline, x, x_prep, weight);
        x_in = x_prep;
    }

    for (uint8_t f = 0; f < tree->params.n_features; ++f) {
        if (tree->params.feature_types[f] == HT_FEAT_NUMERIC) {
            if (!HT_ISNAN(x_in[f].num)) {
                ht_var_update(&tree->feat_stats[f], x_in[f].num, weight);
            }
        }
    }

    tree->train_weight_seen += weight;

    if (tree->params.enable_hat && tree->params.bootstrap_sampling) {
        uint32_t k = ht_poisson_1(&tree->rng_state);
        if (k > 0U) {
            weight *= (ht_float_t)k;
        }
    }

    return ht_tree_learn_one_reg_subtree(tree, tree->root_idx, x_in, y, weight, 0);
}

ht_float_t ht_tree_node_promise(const ht_node_t *node, uint8_t n_classes) {
    if (node == NULL || n_classes == 0) {
        return (ht_float_t)0.0;
    }

    ht_float_t total = (ht_float_t)0.0;
    ht_float_t max_w = (ht_float_t)0.0;

    for (uint8_t c = 0; c < n_classes; ++c) {
        ht_float_t w = node->class_counts[c];
        total += w;
        if (w > max_w) {
            max_w = w;
        }
    }

    if (total <= (ht_float_t)0.0) {
        return (ht_float_t)0.0;
    }

    return total - max_w;
}

ht_status_t ht_tree_deactivate_leaf(ht_tree_t *tree, uint16_t leaf_idx) {
    if (tree == NULL || leaf_idx >= HT_MAX_NODES) {
        return HT_ERR_INVALID_ARG;
    }

    ht_node_t *node = &tree->memory.node_pool[leaf_idx];
    if (!(node->flags & HT_NODE_IN_USE) || !(node->flags & HT_NODE_IS_LEAF)) {
        return HT_ERR_INVALID_ARG;
    }

    if (node->stats_idx == HT_INVALID_IDX) {
        return HT_OK; /* Already inactive */
    }

    ht_stats_free(&tree->memory, node->stats_idx);
    node->stats_idx = HT_INVALID_IDX;

    if (tree->n_active_leaves > 0) {
        tree->n_active_leaves--;
    }
    tree->n_inactive_leaves++;

    return HT_OK;
}

ht_status_t ht_tree_reactivate_leaf(ht_tree_t *tree, uint16_t leaf_idx) {
    if (tree == NULL || leaf_idx >= HT_MAX_NODES) {
        return HT_ERR_INVALID_ARG;
    }

    ht_node_t *node = &tree->memory.node_pool[leaf_idx];
    if (!(node->flags & HT_NODE_IN_USE) || !(node->flags & HT_NODE_IS_LEAF)) {
        return HT_ERR_INVALID_ARG;
    }

    if (node->stats_idx != HT_INVALID_IDX) {
        return HT_OK; /* Already active */
    }

    uint16_t s_idx = ht_stats_alloc(&tree->memory);
    if (s_idx == HT_INVALID_IDX) {
        return HT_ERR_OUT_OF_STATS;
    }

    node->stats_idx = s_idx;

    ht_float_t total = (ht_float_t)0.0;
    for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
        total += node->class_counts[c];
    }
    tree->memory.stats_pool[s_idx].last_split_attempt_at = total;

    tree->n_active_leaves++;
    if (tree->n_inactive_leaves > 0) {
        tree->n_inactive_leaves--;
    }

    return HT_OK;
}

ht_status_t ht_tree_deactivate_least_promising_leaf(ht_tree_t *tree) {
    return ht_tree_deactivate_least_promising_leaf_except(tree, HT_INVALID_IDX);
}

ht_status_t ht_tree_save(const ht_tree_t *tree,
                         void *buffer,
                         size_t buffer_cap,
                         size_t *out_bytes_written,
                         bool include_training_stats) {
    if (tree == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t num_nodes_in_use = 0;
    for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
        if (tree->memory.node_pool[i].flags & HT_NODE_IN_USE) {
            num_nodes_in_use++;
        }
    }

    uint16_t num_stats_in_use = 0;
    if (include_training_stats) {
        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            const ht_node_t *node = &tree->memory.node_pool[i];
            if ((node->flags & HT_NODE_IN_USE) &&
                (node->flags & HT_NODE_IS_LEAF) &&
                (node->stats_idx != HT_INVALID_IDX)) {
                num_stats_in_use++;
            }
        }
    }

    size_t total_size = sizeof(ht_serial_header_t) +
        (size_t)num_nodes_in_use * (sizeof(uint16_t) + sizeof(ht_node_t));
    if (include_training_stats) {
        total_size += (size_t)num_stats_in_use * (sizeof(uint16_t) + sizeof(ht_leaf_stats_t));
    }

    if (out_bytes_written != NULL) {
        *out_bytes_written = total_size;
    }

    if (buffer == NULL) {
        return HT_OK; /* Query size mode */
    }

    if (buffer_cap < total_size) {
        return HT_ERR_INVALID_ARG;
    }

    uint8_t *ptr = (uint8_t *)buffer;
    ht_serial_header_t header;
    memset(&header, 0, sizeof(header));
    header.magic = HT_SERIAL_MAGIC;
    header.version = HT_SERIAL_VERSION;
    header.flags = 0;
#ifdef HT_USE_DOUBLE
    header.flags |= HT_SERIAL_FLAG_DOUBLE;
#endif
    if (include_training_stats) {
        header.flags |= HT_SERIAL_FLAG_STATS;
    }
    if (tree->params.task_type == HT_TASK_REGRESSION) {
        header.flags |= HT_SERIAL_FLAG_REG;
    }
    header.float_size = (uint16_t)sizeof(ht_float_t);
    header.num_nodes_in_use = num_nodes_in_use;
    header.num_stats_in_use = num_stats_in_use;
    header.root_idx = tree->root_idx;
    header.n_active_leaves = tree->n_active_leaves;
    header.n_inactive_leaves = tree->n_inactive_leaves;
    header.n_decision_nodes = tree->n_decision_nodes;
    header.max_depth_seen = tree->max_depth_seen;
    header.params = tree->params;
    header.train_weight_seen = tree->train_weight_seen;

    memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    /* Write in-use nodes */
    for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
        if (tree->memory.node_pool[i].flags & HT_NODE_IN_USE) {
            memcpy(ptr, &i, sizeof(uint16_t));
            ptr += sizeof(uint16_t);
            memcpy(ptr, &tree->memory.node_pool[i], sizeof(ht_node_t));
            ptr += sizeof(ht_node_t);
        }
    }

    /* Write in-use stats if requested */
    if (include_training_stats) {
        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            const ht_node_t *node = &tree->memory.node_pool[i];
            if ((node->flags & HT_NODE_IN_USE) &&
                (node->flags & HT_NODE_IS_LEAF) &&
                (node->stats_idx != HT_INVALID_IDX)) {
                uint16_t s_idx = node->stats_idx;
                memcpy(ptr, &s_idx, sizeof(uint16_t));
                ptr += sizeof(uint16_t);
                memcpy(ptr, &tree->memory.stats_pool[s_idx], sizeof(ht_leaf_stats_t));
                ptr += sizeof(ht_leaf_stats_t);
            }
        }
    }

    return HT_OK;
}

ht_status_t ht_tree_load(ht_tree_t *tree,
                         const void *buffer,
                         size_t buffer_len) {
    if (tree == NULL || buffer == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    if (buffer_len < sizeof(ht_serial_header_t)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    const uint8_t *ptr = (const uint8_t *)buffer;
    ht_serial_header_t header;
    memcpy(&header, ptr, sizeof(header));
    ptr += sizeof(header);

    if (header.magic != HT_SERIAL_MAGIC) {
        return HT_ERR_CORRUPT_IMAGE;
    }
    if (header.version != HT_SERIAL_VERSION) {
        return HT_ERR_CORRUPT_IMAGE;
    }
    if (header.float_size != (uint16_t)sizeof(ht_float_t)) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    size_t expected_size = sizeof(ht_serial_header_t) +
        (size_t)header.num_nodes_in_use * (sizeof(uint16_t) + sizeof(ht_node_t));
    if (header.flags & HT_SERIAL_FLAG_STATS) {
        expected_size += (size_t)header.num_stats_in_use * (sizeof(uint16_t) + sizeof(ht_leaf_stats_t));
    }

    if (buffer_len < expected_size) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    const ht_criterion_vtable_t *vtable = ht_criterion_get_vtable(header.params.criterion);
    if (vtable == NULL) {
        return HT_ERR_CORRUPT_IMAGE;
    }

    memset(tree, 0, sizeof(*tree));
    tree->params = header.params;
    tree->criterion_vtable = vtable;
    tree->root_idx = header.root_idx;
    tree->n_active_leaves = header.n_active_leaves;
    tree->n_inactive_leaves = header.n_inactive_leaves;
    tree->n_decision_nodes = header.n_decision_nodes;
    tree->max_depth_seen = header.max_depth_seen;
    tree->train_weight_seen = header.train_weight_seen;
    tree->rng_state = 123456789U;

    for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
        tree->node_drift[i].adwin_idx = HT_INVALID_IDX;
        tree->node_drift[i].alt_tree_idx = HT_INVALID_IDX;
    }

    /* Read nodes */
    for (uint16_t k = 0; k < header.num_nodes_in_use; ++k) {
        uint16_t node_idx;
        memcpy(&node_idx, ptr, sizeof(uint16_t));
        ptr += sizeof(uint16_t);
        if (node_idx >= HT_MAX_NODES) {
            return HT_ERR_CORRUPT_IMAGE;
        }
        memcpy(&tree->memory.node_pool[node_idx], ptr, sizeof(ht_node_t));
        ptr += sizeof(ht_node_t);
    }

    /* Read stats if present */
    if (header.flags & HT_SERIAL_FLAG_STATS) {
        for (uint16_t k = 0; k < header.num_stats_in_use; ++k) {
            uint16_t stats_idx;
            memcpy(&stats_idx, ptr, sizeof(uint16_t));
            ptr += sizeof(uint16_t);
            if (stats_idx >= HT_MAX_ACTIVE_LEAVES) {
                return HT_ERR_CORRUPT_IMAGE;
            }
            memcpy(&tree->memory.stats_pool[stats_idx], ptr, sizeof(ht_leaf_stats_t));
            ptr += sizeof(ht_leaf_stats_t);
        }
    } else {
        /* When loading inference-only image, ensure all leaves are marked inactive */
        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            if (tree->memory.node_pool[i].flags & HT_NODE_IS_LEAF) {
                tree->memory.node_pool[i].stats_idx = HT_INVALID_IDX;
            }
        }
        tree->n_active_leaves = 0;
        tree->n_inactive_leaves = (header.num_nodes_in_use > header.n_decision_nodes)
            ? (header.num_nodes_in_use - header.n_decision_nodes) : 0;
    }

    /* Reconstruct memory pool free lists */
    tree->memory.free_nodes_top = 0;
    tree->memory.nodes_allocated = 0;
    for (int32_t i = (int32_t)HT_MAX_NODES - 1; i >= 0; --i) {
        if (!(tree->memory.node_pool[i].flags & HT_NODE_IN_USE)) {
            tree->memory.free_nodes[tree->memory.free_nodes_top++] = (uint16_t)i;
        } else {
            tree->memory.nodes_allocated++;
        }
    }

    bool stats_in_use[HT_MAX_ACTIVE_LEAVES];
    memset(stats_in_use, 0, sizeof(stats_in_use));
    if (header.flags & HT_SERIAL_FLAG_STATS) {
        for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
            const ht_node_t *node = &tree->memory.node_pool[i];
            if ((node->flags & HT_NODE_IN_USE) &&
                (node->flags & HT_NODE_IS_LEAF) &&
                (node->stats_idx < HT_MAX_ACTIVE_LEAVES)) {
                stats_in_use[node->stats_idx] = true;
            }
        }
    }

    tree->memory.free_stats_top = 0;
    tree->memory.stats_allocated = 0;
    for (int32_t i = (int32_t)HT_MAX_ACTIVE_LEAVES - 1; i >= 0; --i) {
        if (!stats_in_use[i]) {
            tree->memory.free_stats[tree->memory.free_stats_top++] = (uint16_t)i;
        } else {
            tree->memory.stats_allocated++;
        }
    }

    return HT_OK;
}

const char *ht_version(void) {
    return HT_VERSION_STRING;
}
