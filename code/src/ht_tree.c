/**
 * @file ht_tree.c
 * @brief Complete Hoeffding Tree (VFDT) implementation in pure C11.
 *
 * Implements deterministic routing, multi-strategy leaf predictions (MC, NB, NBA),
 * online learning, memory-bounded leaf deactivation, and binary serialization.
 */

#include "ht_tree.h"
#include <string.h>

/* ========================================================================= */
/* --- Internal Static Helpers --------------------------------------------- */
/* ========================================================================= */

static uint16_t ht_most_common_child(const ht_tree_t *tree, const ht_node_t *node) {
    ht_float_t left_w = (ht_float_t)0.0;
    ht_float_t right_w = (ht_float_t)0.0;

    if (node->left_child != HT_INVALID_IDX) {
        const ht_node_t *left = &tree->memory.node_pool[node->left_child];
        for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
            left_w += left->class_counts[c];
        }
    }
    if (node->right_child != HT_INVALID_IDX) {
        const ht_node_t *right = &tree->memory.node_pool[node->right_child];
        for (uint8_t c = 0; c < tree->params.n_classes; ++c) {
            right_w += right->class_counts[c];
        }
    }

    return (left_w >= right_w) ? node->left_child : node->right_child;
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
                        ht_float_t log_p = ht_gaussian_cond_log_proba(&stats->observers[f].num[c], x[f].num);
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

static ht_status_t ht_tree_deactivate_least_promising_leaf_except(ht_tree_t *tree,
                                                                 uint16_t except_idx) {
    if (tree == NULL || tree->n_active_leaves == 0) {
        return HT_ERR_OUT_OF_STATS;
    }

    uint16_t min_idx = HT_INVALID_IDX;
    ht_float_t min_promise = HT_INFINITY;

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

    if (min_idx == HT_INVALID_IDX) {
        return HT_ERR_OUT_OF_STATS;
    }

    return ht_tree_deactivate_leaf(tree, min_idx);
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

    ht_find_best_split_candidates(stats,
                                  node->class_counts,
                                  tree->params.feature_types,
                                  tree->params.n_features,
                                  tree->params.n_classes,
                                  tree->criterion_vtable,
                                  tree->params.min_branch_fraction,
                                  tree->params.merit_preprune,
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
}

ht_status_t ht_tree_init(ht_tree_t *tree, const ht_config_params_t *params) {
    if (tree == NULL || params == NULL) {
        return HT_ERR_INVALID_ARG;
    }
    if (params->n_features == 0 || params->n_features > HT_MAX_FEATURES) {
        return HT_ERR_INVALID_ARG;
    }
    if (params->n_classes < 2 || params->n_classes > HT_MAX_CLASSES) {
        return HT_ERR_INVALID_ARG;
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

    memset(tree, 0, sizeof(*tree));
    tree->params = *params;
    tree->criterion_vtable = vtable;

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

    tree->root_idx = root;
    tree->n_active_leaves = 1;
    tree->n_inactive_leaves = 0;
    tree->n_decision_nodes = 0;
    tree->train_weight_seen = (ht_float_t)0.0;
    tree->max_depth_seen = 0;

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

ht_status_t ht_tree_predict_proba(const ht_tree_t *tree,
                                  const ht_value_t x[HT_MAX_FEATURES],
                                  ht_float_t proba_out[HT_MAX_CLASSES]) {
    if (tree == NULL || x == NULL || proba_out == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t leaf_idx = ht_tree_route(tree, x, NULL, NULL, NULL);
    if (leaf_idx == HT_INVALID_IDX) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            proba_out[c] = (ht_float_t)0.0;
        }
        return HT_ERR_INVALID_ARG;
    }

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
                    return HT_OK;
                }
            }
        } else if (tree->params.leaf_prediction == HT_LEAF_NBA) {
            if (stats->nb_correct_weight >= stats->mc_correct_weight) {
                if (ht_leaf_predict_nb_proba(tree, node, stats, x, proba_out)) {
                    return HT_OK;
                }
            }
        }
    }

    /* Fallback to Majority Class */
    ht_leaf_predict_mc_proba(node, tree->params.n_classes, proba_out);
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

    tree->train_weight_seen += weight;

    /* Route sample to leaf, updating class_counts along internal nodes */
    uint16_t curr_idx = tree->root_idx;
    uint16_t depth = 0;

    while (curr_idx != HT_INVALID_IDX) {
        ht_node_t *node = &tree->memory.node_pool[curr_idx];
        if (node->flags & HT_NODE_IS_LEAF) {
            break;
        }

        /* Update internal branch class count */
        node->class_counts[y] += weight;

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
        depth++;
    }

    uint16_t leaf_idx = curr_idx;
    ht_node_t *leaf = &tree->memory.node_pool[leaf_idx];

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
                    ht_gaussian_update(stats->observers[f].num, y, x[f].num, weight);
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
