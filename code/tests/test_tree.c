/**
 * @file test_tree.c
 * @brief Level 1 (Unit) and Level 2 (Integration) Tests for Hoeffding Tree (VFDT).
 *
 * Covers:
 * - Default parameters and initialization.
 * - Leaf Promise(l) calculation.
 * - Iterative routing with numeric, categorical, and NaN/missing features (most_common_path).
 * - Multi-strategy leaf prediction: MC, NB (with Log-Sum-Exp), and NBA (prequential tracking).
 * - Incremental training cycle (learn_one), Hoeffding bound split trigger, and stats recycling.
 * - Memory management: leaf deactivation and pool exhaustion handling.
 * - Binary serialization (ht_tree_save / ht_tree_load) with bit-identical parity.
 */

#include "test_harness.h"
#include "ht_tree.h"
#include "ht_prep.h"
#include <string.h>

/* ========================================================================= */
/* --- Level 1: Unit Tests ------------------------------------------------- */
/* ========================================================================= */

static void test_tree_default_params_and_init(void) {
    ht_config_params_t params;
    ht_default_params(&params, 8, 3);

    HT_ASSERT_EQ(params.n_features, 8);
    HT_ASSERT_EQ(params.n_classes, 3);
    HT_ASSERT_EQ(params.criterion, HT_CRIT_INFO_GAIN);
    HT_ASSERT_EQ(params.leaf_prediction, HT_LEAF_NBA);
    HT_ASSERT_EQ(params.grace_period, 200);
    HT_ASSERT_EQ(params.max_depth, 64);
    HT_ASSERT_FLOAT_NEAR(params.delta, (ht_float_t)1e-7, 1e-12);
    HT_ASSERT_FLOAT_NEAR(params.tau, (ht_float_t)0.05, 1e-6);
    HT_ASSERT_FLOAT_NEAR(params.min_branch_fraction, (ht_float_t)0.01, 1e-6);
    HT_ASSERT_FLOAT_NEAR(params.max_share_to_split, (ht_float_t)0.99, 1e-6);
    HT_ASSERT(params.merit_preprune);

    /* Initialization validation */
    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(NULL, &params), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_tree_init(&tree, NULL), HT_ERR_INVALID_ARG);

    /* Invalid n_features / n_classes */
    ht_config_params_t bad_params = params;
    bad_params.n_features = 0;
    HT_ASSERT_EQ(ht_tree_init(&tree, &bad_params), HT_ERR_INVALID_ARG);
    bad_params.n_features = HT_MAX_FEATURES + 1;
    HT_ASSERT_EQ(ht_tree_init(&tree, &bad_params), HT_ERR_INVALID_ARG);

    bad_params = params;
    bad_params.n_classes = 1;
    HT_ASSERT_EQ(ht_tree_init(&tree, &bad_params), HT_ERR_INVALID_ARG);

    /* Clean init */
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);
    HT_ASSERT_EQ(tree.root_idx, 0);
    HT_ASSERT_EQ(tree.n_active_leaves, 1);
    HT_ASSERT_EQ(tree.n_inactive_leaves, 0);
    HT_ASSERT_EQ(tree.n_decision_nodes, 0);
    HT_ASSERT_EQ(tree.max_depth_seen, 0);
    HT_ASSERT_FLOAT_EQ(tree.train_weight_seen, (ht_float_t)0.0);

    const ht_node_t *root = &tree.memory.node_pool[tree.root_idx];
    HT_ASSERT(root->flags & HT_NODE_IS_LEAF);
    HT_ASSERT(root->flags & HT_NODE_IN_USE);
    HT_ASSERT_NEQ(root->stats_idx, HT_INVALID_IDX);
}

static void test_tree_node_promise_analytical(void) {
    ht_node_t node;
    memset(&node, 0, sizeof(node));

    /* Empty node: promise = 0.0 */
    HT_ASSERT_FLOAT_EQ(ht_tree_node_promise(&node, 3), (ht_float_t)0.0);

    /* Pure node [100, 0, 0]: total = 100, max = 100 -> promise = 0.0 */
    node.class_counts[0] = (ht_float_t)100.0;
    HT_ASSERT_FLOAT_EQ(ht_tree_node_promise(&node, 3), (ht_float_t)0.0);

    /* Balanced node [30, 30, 30]: total = 90, max = 30 -> promise = 60.0 */
    node.class_counts[1] = (ht_float_t)30.0;
    node.class_counts[2] = (ht_float_t)30.0;
    node.class_counts[0] = (ht_float_t)30.0;
    HT_ASSERT_FLOAT_EQ(ht_tree_node_promise(&node, 3), (ht_float_t)60.0);

    /* Skewed node [70, 20, 10]: total = 100, max = 70 -> promise = 30.0 */
    node.class_counts[0] = (ht_float_t)70.0;
    node.class_counts[1] = (ht_float_t)20.0;
    node.class_counts[2] = (ht_float_t)10.0;
    HT_ASSERT_FLOAT_EQ(ht_tree_node_promise(&node, 3), (ht_float_t)30.0);
}

static void test_tree_routing_numeric_and_categorical(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 4, 2);
    params.feature_types[0] = HT_FEAT_NUMERIC;
    params.feature_types[1] = HT_FEAT_CATEGORICAL;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    /* Manually build a 2-level test tree:
     * Node 0 (root): split on numeric feature 0 <= 5.0
     *   Left child -> Node 1: split on categorical feature 1 == 2
     *     Left child -> Node 3 (leaf: left-left)
     *     Right child -> Node 4 (leaf: left-right)
     *   Right child -> Node 2 (leaf: right)
     */
    uint16_t n1 = ht_node_alloc(&tree.memory);
    uint16_t n2 = ht_node_alloc(&tree.memory);
    uint16_t n3 = ht_node_alloc(&tree.memory);
    uint16_t n4 = ht_node_alloc(&tree.memory);

    ht_node_t *r = &tree.memory.node_pool[tree.root_idx];
    r->flags = HT_NODE_IN_USE; /* internal */
    r->feature_idx = 0;
    r->split_val.threshold = (ht_float_t)5.0;
    r->left_child = n1;
    r->right_child = n2;

    ht_node_t *node1 = &tree.memory.node_pool[n1];
    node1->flags = HT_NODE_IN_USE | HT_NODE_SPLIT_CAT; /* internal categorical */
    node1->feature_idx = 1;
    node1->split_val.cat_val = 2;
    node1->left_child = n3;
    node1->right_child = n4;

    tree.memory.node_pool[n2].flags = HT_NODE_IN_USE | HT_NODE_IS_LEAF;
    tree.memory.node_pool[n3].flags = HT_NODE_IN_USE | HT_NODE_IS_LEAF;
    tree.memory.node_pool[n4].flags = HT_NODE_IN_USE | HT_NODE_IS_LEAF;

    /* Route test 1: x[0] = 3.0 (<= 5.0), x[1] = 2 (== 2) -> reaches n3 at depth 2 */
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    x[0].num = (ht_float_t)3.0;
    x[1].cat = 2;

    uint16_t parent = HT_INVALID_IDX;
    uint8_t branch = 255;
    uint16_t depth = 999;
    uint16_t dest = ht_tree_route(&tree, x, &parent, &branch, &depth);
    HT_ASSERT_EQ(dest, n3);
    HT_ASSERT_EQ(parent, n1);
    HT_ASSERT_EQ(branch, 0);
    HT_ASSERT_EQ(depth, 2);

    /* Route test 2: x[0] = 3.0 (<= 5.0), x[1] = 0 (!= 2) -> reaches n4 at depth 2 */
    x[1].cat = 0;
    dest = ht_tree_route(&tree, x, &parent, &branch, &depth);
    HT_ASSERT_EQ(dest, n4);
    HT_ASSERT_EQ(parent, n1);
    HT_ASSERT_EQ(branch, 1);
    HT_ASSERT_EQ(depth, 2);

    /* Route test 3: x[0] = 8.5 (> 5.0) -> reaches n2 at depth 1 */
    x[0].num = (ht_float_t)8.5;
    dest = ht_tree_route(&tree, x, &parent, &branch, &depth);
    HT_ASSERT_EQ(dest, n2);
    HT_ASSERT_EQ(parent, tree.root_idx);
    HT_ASSERT_EQ(branch, 1);
    HT_ASSERT_EQ(depth, 1);
}

static void test_tree_routing_missing_nan_and_most_common_path(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    params.feature_types[0] = HT_FEAT_NUMERIC;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    uint16_t left_child = ht_node_alloc(&tree.memory);
    uint16_t right_child = ht_node_alloc(&tree.memory);

    ht_node_t *root = &tree.memory.node_pool[tree.root_idx];
    root->flags = HT_NODE_IN_USE; /* internal */
    root->feature_idx = 0;
    root->split_val.threshold = (ht_float_t)10.0;
    root->left_child = left_child;
    root->right_child = right_child;

    ht_node_t *left = &tree.memory.node_pool[left_child];
    left->flags = HT_NODE_IN_USE | HT_NODE_IS_LEAF;
    left->class_counts[0] = (ht_float_t)70.0;
    left->class_counts[1] = (ht_float_t)30.0; /* Left sum = 100 */

    ht_node_t *right = &tree.memory.node_pool[right_child];
    right->flags = HT_NODE_IN_USE | HT_NODE_IS_LEAF;
    right->class_counts[0] = (ht_float_t)10.0;
    right->class_counts[1] = (ht_float_t)20.0; /* Right sum = 30 */

    /* Query with NaN: left child has more weight (100 >= 30), must pick left */
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    x[0].num = (ht_float_t)(0.0 / 0.0); /* NaN */

    uint16_t dest = ht_tree_route(&tree, x, NULL, NULL, NULL);
    HT_ASSERT_EQ(dest, left_child);

    /* Update right child weight to dominate (120 > 100) */
    right->class_counts[0] = (ht_float_t)100.0;
    right->class_counts[1] = (ht_float_t)20.0;

    dest = ht_tree_route(&tree, x, NULL, NULL, NULL);
    HT_ASSERT_EQ(dest, right_child);
}

static void test_tree_predict_majority_class(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 2, 3);
    params.leaf_prediction = HT_LEAF_MC;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_node_t *root = &tree.memory.node_pool[tree.root_idx];
    root->class_counts[0] = (ht_float_t)10.0;
    root->class_counts[1] = (ht_float_t)60.0;
    root->class_counts[2] = (ht_float_t)30.0;

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    x[0].num = (ht_float_t)1.0;

    ht_float_t probas[HT_MAX_CLASSES];
    HT_ASSERT_EQ(ht_tree_predict_proba(&tree, x, probas), HT_OK);

    HT_ASSERT_FLOAT_NEAR(probas[0], (ht_float_t)0.10, 1e-6);
    HT_ASSERT_FLOAT_NEAR(probas[1], (ht_float_t)0.60, 1e-6);
    HT_ASSERT_FLOAT_NEAR(probas[2], (ht_float_t)0.30, 1e-6);
    HT_ASSERT_FLOAT_EQ(probas[3], (ht_float_t)0.0);

    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 1);
}

static void test_tree_predict_naive_bayes_lse(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    params.leaf_prediction = HT_LEAF_NB;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_node_t *root = &tree.memory.node_pool[tree.root_idx];
    root->class_counts[0] = (ht_float_t)50.0;
    root->class_counts[1] = (ht_float_t)50.0;

    ht_leaf_stats_t *stats = &tree.memory.stats_pool[root->stats_idx];

    /* Populate Class 0: clustered around (1.0, 1.0) */
    for (int i = 0; i < 50; ++i) {
        ht_gaussian_update(stats->observers[0].num, 0, (ht_float_t)(0.8 + 0.008 * i), (ht_float_t)1.0);
        ht_gaussian_update(stats->observers[1].num, 0, (ht_float_t)(0.9 + 0.004 * i), (ht_float_t)1.0);
    }

    /* Populate Class 1: clustered around (10.0, 10.0) */
    for (int i = 0; i < 50; ++i) {
        ht_gaussian_update(stats->observers[0].num, 1, (ht_float_t)(9.5 + 0.02 * i), (ht_float_t)1.0);
        ht_gaussian_update(stats->observers[1].num, 1, (ht_float_t)(10.0 + 0.01 * i), (ht_float_t)1.0);
    }

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    ht_float_t probas[HT_MAX_CLASSES];

    /* Query point near Class 0 cluster: (1.0, 1.0) */
    x[0].num = (ht_float_t)1.0;
    x[1].num = (ht_float_t)1.0;
    HT_ASSERT_EQ(ht_tree_predict_proba(&tree, x, probas), HT_OK);
    HT_ASSERT(probas[0] > (ht_float_t)0.99);
    HT_ASSERT_FLOAT_NEAR(probas[0] + probas[1], (ht_float_t)1.0, 1e-5);
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 0);

    /* Query point near Class 1 cluster: (10.0, 10.0) */
    x[0].num = (ht_float_t)10.0;
    x[1].num = (ht_float_t)10.0;
    HT_ASSERT_EQ(ht_tree_predict_proba(&tree, x, probas), HT_OK);
    HT_ASSERT(probas[1] > (ht_float_t)0.99);
    HT_ASSERT_FLOAT_NEAR(probas[0] + probas[1], (ht_float_t)1.0, 1e-5);
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 1);
}

static void test_tree_predict_nba_adaptive_switching(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 1, 2);
    params.leaf_prediction = HT_LEAF_NBA;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_node_t *root = &tree.memory.node_pool[tree.root_idx];
    /* MC predicts class 0 overwhelmingly */
    root->class_counts[0] = (ht_float_t)90.0;
    root->class_counts[1] = (ht_float_t)10.0;

    ht_leaf_stats_t *stats = &tree.memory.stats_pool[root->stats_idx];

    /* Setup Gaussian observers such that x=10 has very high density in Class 1 */
    for (int i = 0; i < 50; ++i) {
        ht_gaussian_update(stats->observers[0].num, 0, (ht_float_t)(0.0 + 0.01 * i), (ht_float_t)1.0);
        ht_gaussian_update(stats->observers[0].num, 1, (ht_float_t)(10.0 + 0.01 * i), (ht_float_t)1.0);
    }

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    x[0].num = (ht_float_t)10.0;

    /* Case A: nb_correct_weight >= mc_correct_weight -> Uses NB -> predicts Class 1 */
    stats->nb_correct_weight = (ht_float_t)50.0;
    stats->mc_correct_weight = (ht_float_t)30.0;
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 1);

    /* Case B: nb_correct_weight < mc_correct_weight -> Falls back to MC -> predicts Class 0 */
    stats->nb_correct_weight = (ht_float_t)20.0;
    stats->mc_correct_weight = (ht_float_t)50.0;
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 0);
}

/* ========================================================================= */
/* --- Level 2: Integration Tests ------------------------------------------ */
/* ========================================================================= */

static void test_tree_learn_one_and_split(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    params.grace_period = 25;
    params.delta = (ht_float_t)1e-4;
    params.tau = (ht_float_t)0.05;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    HT_ASSERT_EQ(tree.n_decision_nodes, 0);
    HT_ASSERT_EQ(tree.n_active_leaves, 1);

    /* Stream clearly separable samples on feature 0:
     * Class 0: x[0] in [0.0, 4.0]
     * Class 1: x[0] in [6.0, 10.0]
     */
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 200; ++i) {
        if (i % 2 == 0) {
            x[0].num = (ht_float_t)(1.0 + (i % 3));
            x[1].num = (ht_float_t)0.5;
            HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, 0, (ht_float_t)1.0), HT_OK);
        } else {
            x[0].num = (ht_float_t)(7.0 + (i % 3));
            x[1].num = (ht_float_t)0.5;
            HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, 1, (ht_float_t)1.0), HT_OK);
        }
    }

    /* Tree should have triggered a split on feature 0 */
    HT_ASSERT(tree.n_decision_nodes >= 1);
    HT_ASSERT(tree.n_active_leaves >= 2);
    HT_ASSERT(tree.max_depth_seen >= 1);

    const ht_node_t *root = &tree.memory.node_pool[tree.root_idx];
    HT_ASSERT(!(root->flags & HT_NODE_IS_LEAF)); /* Root is now an internal branch */
    HT_ASSERT_EQ(root->feature_idx, 0);          /* Split on feature 0 */
    HT_ASSERT_EQ(root->stats_idx, HT_INVALID_IDX); /* Root stats recycled! */

    /* The split threshold must be between 4.0 and 6.0 */
    HT_ASSERT(root->split_val.threshold >= (ht_float_t)3.0 &&
              root->split_val.threshold <= (ht_float_t)7.0);

    /* Test that new samples route accurately through split */
    x[0].num = (ht_float_t)2.0;
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 0);

    x[0].num = (ht_float_t)8.0;
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 1);
}

static void test_tree_leaf_deactivation_and_reactivation(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    uint16_t root_idx = tree.root_idx;
    HT_ASSERT_EQ(tree.n_active_leaves, 1);
    HT_ASSERT_EQ(tree.n_inactive_leaves, 0);

    /* Deactivate leaf */
    HT_ASSERT_EQ(ht_tree_deactivate_leaf(&tree, root_idx), HT_OK);
    HT_ASSERT_EQ(tree.n_active_leaves, 0);
    HT_ASSERT_EQ(tree.n_inactive_leaves, 1);
    HT_ASSERT_EQ(tree.memory.node_pool[root_idx].stats_idx, HT_INVALID_IDX);

    /* Double deactivation is a safe no-op */
    HT_ASSERT_EQ(ht_tree_deactivate_leaf(&tree, root_idx), HT_OK);
    HT_ASSERT_EQ(tree.n_active_leaves, 0);
    HT_ASSERT_EQ(tree.n_inactive_leaves, 1);

    /* Reactivate leaf */
    HT_ASSERT_EQ(ht_tree_reactivate_leaf(&tree, root_idx), HT_OK);
    HT_ASSERT_EQ(tree.n_active_leaves, 1);
    HT_ASSERT_EQ(tree.n_inactive_leaves, 0);
    HT_ASSERT_NEQ(tree.memory.node_pool[root_idx].stats_idx, HT_INVALID_IDX);
}

static void test_tree_leaf_deactivation_least_promising_selection(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    /* Allocate 3 active leaves manually in tree */
    uint16_t leaf1 = tree.root_idx;
    uint16_t leaf2 = ht_node_alloc(&tree.memory);
    uint16_t leaf3 = ht_node_alloc(&tree.memory);

    uint16_t s2 = ht_stats_alloc(&tree.memory);
    uint16_t s3 = ht_stats_alloc(&tree.memory);

    tree.memory.node_pool[leaf2].stats_idx = s2;
    tree.memory.node_pool[leaf2].flags = HT_NODE_IN_USE | HT_NODE_IS_LEAF;
    tree.memory.node_pool[leaf3].stats_idx = s3;
    tree.memory.node_pool[leaf3].flags = HT_NODE_IN_USE | HT_NODE_IS_LEAF;

    tree.n_active_leaves = 3;

    /* leaf1: [50, 50] -> promise = 100 - 50 = 50.0 */
    tree.memory.node_pool[leaf1].class_counts[0] = (ht_float_t)50.0;
    tree.memory.node_pool[leaf1].class_counts[1] = (ht_float_t)50.0;

    /* leaf2: [95, 5] -> promise = 100 - 95 = 5.0 (Lowest Promise!) */
    tree.memory.node_pool[leaf2].class_counts[0] = (ht_float_t)95.0;
    tree.memory.node_pool[leaf2].class_counts[1] = (ht_float_t)5.0;

    /* leaf3: [70, 30] -> promise = 100 - 70 = 30.0 */
    tree.memory.node_pool[leaf3].class_counts[0] = (ht_float_t)70.0;
    tree.memory.node_pool[leaf3].class_counts[1] = (ht_float_t)30.0;

    /* Deactivating least promising must target leaf2 */
    HT_ASSERT_EQ(ht_tree_deactivate_least_promising_leaf(&tree), HT_OK);
    HT_ASSERT_EQ(tree.memory.node_pool[leaf2].stats_idx, HT_INVALID_IDX);
    HT_ASSERT_NEQ(tree.memory.node_pool[leaf1].stats_idx, HT_INVALID_IDX);
    HT_ASSERT_NEQ(tree.memory.node_pool[leaf3].stats_idx, HT_INVALID_IDX);
    HT_ASSERT_EQ(tree.n_active_leaves, 2);
    HT_ASSERT_EQ(tree.n_inactive_leaves, 1);
}

static void test_tree_leaf_deactivation_pool_exhaustion_stress(void) {
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    params.grace_period = 20;
    params.delta = (ht_float_t)1e-3;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    /* Artificially exhaust stats_pool by allocating all remaining stats blocks */
    uint16_t dummy_stats[HT_MAX_ACTIVE_LEAVES];
    uint16_t dummy_count = 0;
    while (true) {
        uint16_t idx = ht_stats_alloc(&tree.memory);
        if (idx == HT_INVALID_IDX) break;
        dummy_stats[dummy_count++] = idx;
    }
    HT_ASSERT_EQ(ht_stats_available(&tree.memory), 0);

    /* Stream separable samples to trigger split */
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 150; ++i) {
        if (i % 2 == 0) {
            x[0].num = (ht_float_t)1.0;
            HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, 0, (ht_float_t)1.0), HT_OK);
        } else {
            x[0].num = (ht_float_t)9.0;
            HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, 1, (ht_float_t)1.0), HT_OK);
        }
    }

    /* Split must succeed without crash despite stats pool exhaustion */
    HT_ASSERT(tree.n_decision_nodes >= 1);

    /* Cleanup dummy allocations */
    for (uint16_t i = 0; i < dummy_count; ++i) {
        ht_stats_free(&tree.memory, dummy_stats[i]);
    }
}

static void test_tree_serialization_roundtrip_full_and_inference(void) {
    ht_tree_t original_tree;
    ht_config_params_t params;
    ht_default_params(&params, 3, 2);
    params.grace_period = 25;
    params.delta = (ht_float_t)1e-4;
    HT_ASSERT_EQ(ht_tree_init(&original_tree, &params), HT_OK);

    /* Train tree to create some decision nodes and leaves */
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 200; ++i) {
        x[0].num = (i % 2 == 0) ? (ht_float_t)1.5 : (ht_float_t)8.5;
        x[1].num = (ht_float_t)(i * 0.1);
        x[2].cat = i % 3;
        uint8_t y = (i % 2 == 0) ? 0 : 1;
        HT_ASSERT_EQ(ht_tree_learn_one(&original_tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    /* Buffer capacity query */
    size_t full_bytes_needed = 0;
    HT_ASSERT_EQ(ht_tree_save(&original_tree, NULL, 0, &full_bytes_needed, true), HT_OK);
    HT_ASSERT(full_bytes_needed > sizeof(ht_serial_header_t));

    size_t infer_bytes_needed = 0;
    HT_ASSERT_EQ(ht_tree_save(&original_tree, NULL, 0, &infer_bytes_needed, false), HT_OK);
    HT_ASSERT(infer_bytes_needed < full_bytes_needed);

    /* Save full model with training stats */
    uint8_t full_buffer[65536];
    HT_ASSERT(sizeof(full_buffer) >= full_bytes_needed);
    size_t written = 0;
    HT_ASSERT_EQ(ht_tree_save(&original_tree, full_buffer, sizeof(full_buffer), &written, true), HT_OK);
    HT_ASSERT_EQ(written, full_bytes_needed);

    /* Load into fresh tree */
    ht_tree_t loaded_full;
    HT_ASSERT_EQ(ht_tree_load(&loaded_full, full_buffer, written), HT_OK);

    /* Verify metadata equality */
    HT_ASSERT_EQ(loaded_full.root_idx, original_tree.root_idx);
    HT_ASSERT_EQ(loaded_full.n_decision_nodes, original_tree.n_decision_nodes);
    HT_ASSERT_EQ(loaded_full.n_active_leaves, original_tree.n_active_leaves);
    HT_ASSERT_EQ(loaded_full.n_inactive_leaves, original_tree.n_inactive_leaves);
    HT_ASSERT_FLOAT_EQ(loaded_full.train_weight_seen, original_tree.train_weight_seen);

    /* Save inference-only model (no training stats) */
    uint8_t infer_buffer[16384];
    HT_ASSERT(sizeof(infer_buffer) >= infer_bytes_needed);
    HT_ASSERT_EQ(ht_tree_save(&original_tree, infer_buffer, sizeof(infer_buffer), &written, false), HT_OK);
    HT_ASSERT_EQ(written, infer_bytes_needed);

    /* Load inference-only model into fresh tree */
    ht_tree_t loaded_infer;
    HT_ASSERT_EQ(ht_tree_load(&loaded_infer, infer_buffer, written), HT_OK);
    HT_ASSERT_EQ(loaded_infer.n_active_leaves, 0);
    HT_ASSERT(loaded_infer.n_inactive_leaves > 0);

    /* Verify prediction parity across all models on test points */
    for (int t = 0; t < 20; ++t) {
        x[0].num = (ht_float_t)(t * 0.5);
        x[1].num = (ht_float_t)(t * 0.2);
        x[2].cat = t % 3;

        ht_float_t proba_orig[HT_MAX_CLASSES];
        ht_float_t proba_full[HT_MAX_CLASSES];
        ht_float_t proba_infer[HT_MAX_CLASSES];

        HT_ASSERT_EQ(ht_tree_predict_proba(&original_tree, x, proba_orig), HT_OK);
        HT_ASSERT_EQ(ht_tree_predict_proba(&loaded_full, x, proba_full), HT_OK);
        HT_ASSERT_EQ(ht_tree_predict_proba(&loaded_infer, x, proba_infer), HT_OK);

        /* Loaded full model must match bit-for-bit with original */
        HT_ASSERT_FLOAT_NEAR(proba_orig[0], proba_full[0], 1e-6);
        HT_ASSERT_FLOAT_NEAR(proba_orig[1], proba_full[1], 1e-6);

        HT_ASSERT_EQ(ht_tree_predict(&original_tree, x), ht_tree_predict(&loaded_full, x));
    }

    /* Test corrupt buffer rejection */
    uint8_t corrupt_buffer[128];
    memset(corrupt_buffer, 0, sizeof(corrupt_buffer));
    HT_ASSERT_EQ(ht_tree_load(&loaded_full, corrupt_buffer, sizeof(corrupt_buffer)), HT_ERR_CORRUPT_IMAGE);
    HT_ASSERT_EQ(ht_tree_load(&loaded_full, full_buffer, sizeof(ht_serial_header_t) - 1), HT_ERR_CORRUPT_IMAGE);
}

/* ========================================================================= */
/* --- Milestone 4.4: Regression Trees (VFDT-R & HAT-R) Tests -------------- */
/* ========================================================================= */

static void test_tree_reg_default_params_and_init(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 4);

    HT_ASSERT_EQ(params.task_type, HT_TASK_REGRESSION);
    HT_ASSERT_EQ(params.n_features, 4U);
    HT_ASSERT_EQ(params.n_classes, 1U);
    HT_ASSERT_EQ(params.criterion, HT_CRIT_VARIANCE_REDUCTION);
    HT_ASSERT_EQ(params.leaf_prediction, HT_LEAF_ADAPTIVE);
    HT_ASSERT_EQ(params.grace_period, 200U);
    HT_ASSERT_FLOAT_NEAR(params.delta, (ht_float_t)1e-7, (ht_float_t)1e-12);
    HT_ASSERT_FLOAT_NEAR(params.tau, (ht_float_t)0.05, (ht_float_t)1e-6);
    HT_ASSERT_FLOAT_NEAR(params.learning_rate, (ht_float_t)0.01, (ht_float_t)1e-6);
    HT_ASSERT_FLOAT_NEAR(params.l2_regularization, (ht_float_t)1e-4, (ht_float_t)1e-8);
    HT_ASSERT_FLOAT_NEAR(params.model_selector_decay, (ht_float_t)0.95, (ht_float_t)1e-6);
    HT_ASSERT_FLOAT_NEAR(params.min_samples_split, (ht_float_t)5.0, (ht_float_t)1e-6);
    HT_ASSERT(!params.enable_hat);

    ht_config_params_t hat_params;
    ht_default_hat_reg_params(&hat_params, 4);
    HT_ASSERT(hat_params.enable_hat);
    HT_ASSERT(hat_params.bootstrap_sampling);

    /* Init tree */
    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);
    HT_ASSERT_EQ(tree.root_idx, 0U);
    HT_ASSERT_EQ(tree.n_active_leaves, 1U);
    HT_ASSERT_EQ(tree.n_decision_nodes, 0U);
    HT_ASSERT_FLOAT_EQ(tree.memory.node_pool[0].reg_stats.count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(tree.memory.node_pool[0].reg_stats.mean, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(tree.memory.node_pool[0].reg_stats.m2, (ht_float_t)0.0);
}

static void test_tree_reg_predict_leaf_mean(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.leaf_prediction = HT_LEAF_MEAN;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Empty tree prediction is 0.0 */
    HT_ASSERT_FLOAT_EQ(ht_tree_predict_reg(&tree, x), (ht_float_t)0.0);

    /* Feed 3 samples: y = 10, 20, 30 */
    x[0].num = (ht_float_t)1.0;
    HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, (ht_float_t)10.0, (ht_float_t)1.0), HT_OK);
    HT_ASSERT_FLOAT_NEAR(ht_tree_predict_reg(&tree, x), (ht_float_t)10.0, (ht_float_t)1e-5);

    HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, (ht_float_t)20.0, (ht_float_t)1.0), HT_OK);
    HT_ASSERT_FLOAT_NEAR(ht_tree_predict_reg(&tree, x), (ht_float_t)15.0, (ht_float_t)1e-5);

    HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, (ht_float_t)30.0, (ht_float_t)1.0), HT_OK);
    HT_ASSERT_FLOAT_NEAR(ht_tree_predict_reg(&tree, x), (ht_float_t)20.0, (ht_float_t)1e-5);

    /* Deactivate leaf: prediction remains target mean (20.0) */
    HT_ASSERT_EQ(ht_tree_deactivate_leaf(&tree, tree.root_idx), HT_OK);
    HT_ASSERT_FLOAT_NEAR(ht_tree_predict_reg(&tree, x), (ht_float_t)20.0, (ht_float_t)1e-5);
}

static void test_tree_reg_predict_leaf_model_sgd_and_clipping(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.leaf_prediction = HT_LEAF_MODEL;
    params.learning_rate = (ht_float_t)0.05;
    params.l2_regularization = (ht_float_t)1e-4;
    params.grace_period = 1000U; /* Retain leaf without splitting during SGD tests */

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Train on linear target y = 3.0 * x_0 + 2.0 */
    for (int i = 0; i < 300; ++i) {
        ht_float_t x_val = (ht_float_t)(i % 10) * (ht_float_t)0.5;
        ht_float_t y_val = (ht_float_t)3.0 * x_val + (ht_float_t)2.0;
        x[0].num = x_val;
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, y_val, (ht_float_t)1.0), HT_OK);
    }

    /* Test linear model prediction on unseen points */
    x[0].num = (ht_float_t)2.0; /* Expected ~ 8.0 */
    ht_float_t pred = ht_tree_predict_reg(&tree, x);
    HT_ASSERT_FLOAT_NEAR(pred, (ht_float_t)8.0, (ht_float_t)1.0);

    /* Test gradient clipping with extreme values */
    x[0].num = (ht_float_t)1e8;
    HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, (ht_float_t)1e14, (ht_float_t)1.0), HT_OK);
    uint16_t s_idx = tree.memory.node_pool[tree.root_idx].stats_idx;
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_weights[0] <= (ht_float_t)1e12);
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_weights[0] >= (ht_float_t)-1e12);
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_intercept <= (ht_float_t)1e12);
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_intercept >= (ht_float_t)-1e12);
}

static void test_tree_reg_predict_leaf_adaptive_fmse(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.leaf_prediction = HT_LEAF_ADAPTIVE;
    params.model_selector_decay = (ht_float_t)0.95;
    params.grace_period = 1000U; /* Retain leaf without splitting during adaptive selector tests */

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* 1. Feed strong linear relationship y = 5.0 * x_0 */
    for (int i = 0; i < 200; ++i) {
        ht_float_t x_val = (ht_float_t)(i % 5);
        ht_float_t y_val = (ht_float_t)5.0 * x_val;
        x[0].num = x_val;
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, y_val, (ht_float_t)1.0), HT_OK);
    }

    uint16_t s_idx = tree.memory.node_pool[tree.root_idx].stats_idx;
    ht_leaf_stats_t *st = &tree.memory.stats_pool[s_idx];

    /* For linear stream, model error is smaller than mean error */
    HT_ASSERT(st->fmse_model < st->fmse_mean);

    /* Adaptive predicts using linear model */
    x[0].num = (ht_float_t)2.0; /* Expected ~ 10.0 */
    HT_ASSERT_FLOAT_NEAR(ht_tree_predict_reg(&tree, x), (ht_float_t)10.0, (ht_float_t)1.5);
}

static void test_tree_reg_split_and_weight_inheritance(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.grace_period = 25U;
    params.delta = (ht_float_t)1e-3;
    params.min_samples_split = (ht_float_t)5.0;
    params.min_branch_fraction = (ht_float_t)0.01;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Train on step function:
     * If x_0 <= 5.0: y ~ 10.0
     * If x_0 > 5.0:  y ~ 100.0
     */
    for (int i = 0; i < 200; ++i) {
        ht_float_t x_val = (ht_float_t)(i % 10);
        ht_float_t y_val = (x_val <= (ht_float_t)4.5) ? (ht_float_t)10.0 : (ht_float_t)100.0;
        x[0].num = x_val;
        x[1].num = (ht_float_t)1.0;
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, y_val, (ht_float_t)1.0), HT_OK);
        if (!(tree.memory.node_pool[tree.root_idx].flags & HT_NODE_IS_LEAF)) {
            break; /* Stop immediately after root split to inspect children */
        }
    }

    /* Verify split occurred at root */
    ht_node_t *root = &tree.memory.node_pool[tree.root_idx];
    HT_ASSERT(!(root->flags & HT_NODE_IS_LEAF));
    HT_ASSERT_EQ(root->feature_idx, 0U);
    HT_ASSERT_NEQ(root->left_child, HT_INVALID_IDX);
    HT_ASSERT_NEQ(root->right_child, HT_INVALID_IDX);
    HT_ASSERT_EQ(root->stats_idx, HT_INVALID_IDX); /* Recycled! */

    /* Verify children inherited SGD weights and fmse */
    ht_node_t *left = &tree.memory.node_pool[root->left_child];
    ht_node_t *right = &tree.memory.node_pool[root->right_child];
    HT_ASSERT(left->flags & HT_NODE_IS_LEAF);
    HT_ASSERT(right->flags & HT_NODE_IS_LEAF);
    HT_ASSERT_NEQ(left->stats_idx, HT_INVALID_IDX);
    HT_ASSERT_NEQ(right->stats_idx, HT_INVALID_IDX);

    /* Left child target mean should be low (< 40), Right child target mean high (> 60) */
    HT_ASSERT(left->reg_stats.mean < (ht_float_t)40.0);
    HT_ASSERT(right->reg_stats.mean > (ht_float_t)60.0);

    /* Test predictions follow the split */
    x[0].num = (ht_float_t)2.0;
    HT_ASSERT(ht_tree_predict_reg(&tree, x) < (ht_float_t)40.0);

    x[0].num = (ht_float_t)8.0;
    HT_ASSERT(ht_tree_predict_reg(&tree, x) > (ht_float_t)70.0);
}

static void test_tree_reg_leaf_deactivation_by_depth(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.grace_period = 20U;
    params.delta = (ht_float_t)1e-2;
    params.min_samples_split = (ht_float_t)5.0;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Create splits to produce leaves at depth 1 and depth 2 */
    for (int i = 0; i < 200; ++i) {
        ht_float_t x0 = (ht_float_t)(i % 10);
        ht_float_t y = (x0 < (ht_float_t)3.0) ? (ht_float_t)5.0 :
                       (x0 < (ht_float_t)7.0) ? (ht_float_t)50.0 : (ht_float_t)200.0;
        x[0].num = x0;
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    uint16_t active_before = tree.n_active_leaves;
    HT_ASSERT(active_before >= 2);

    /* Deactivate least promising leaf */
    HT_ASSERT_EQ(ht_tree_deactivate_least_promising_leaf(&tree), HT_OK);
    HT_ASSERT_EQ(tree.n_active_leaves, (uint16_t)(active_before - 1));
    HT_ASSERT_EQ(tree.n_inactive_leaves, 1U);
}

static void test_tree_reg_serialization_roundtrip(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 3);
    params.grace_period = 30U;
    params.delta = (ht_float_t)1e-2;

    ht_tree_t orig_tree;
    HT_ASSERT_EQ(ht_tree_init(&orig_tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 150; ++i) {
        x[0].num = (ht_float_t)(i % 10);
        x[1].num = (ht_float_t)(i % 5);
        x[2].num = (ht_float_t)(i * 0.1);
        ht_float_t y = (ht_float_t)2.5 * x[0].num + (ht_float_t)1.5 * x[1].num;
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&orig_tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    /* Save full model */
    size_t written = 0;
    HT_ASSERT_EQ(ht_tree_save(&orig_tree, NULL, 0, &written, true), HT_OK);
    uint8_t full_buf[32768];
    HT_ASSERT(sizeof(full_buf) >= written);
    HT_ASSERT_EQ(ht_tree_save(&orig_tree, full_buf, sizeof(full_buf), &written, true), HT_OK);

    /* Load into fresh tree */
    ht_tree_t loaded_tree;
    HT_ASSERT_EQ(ht_tree_load(&loaded_tree, full_buf, written), HT_OK);
    HT_ASSERT_EQ(loaded_tree.params.task_type, HT_TASK_REGRESSION);

    /* Verify prediction equality */
    for (int i = 0; i < 10; ++i) {
        x[0].num = (ht_float_t)(i * 0.7);
        x[1].num = (ht_float_t)(i * 0.3);
        x[2].num = (ht_float_t)(i * 0.1);
        ht_float_t p_orig = ht_tree_predict_reg(&orig_tree, x);
        ht_float_t p_load = ht_tree_predict_reg(&loaded_tree, x);
        HT_ASSERT_FLOAT_NEAR(p_orig, p_load, (ht_float_t)1e-5);
    }
}

static void test_tree_hat_reg_alternate_tree_switch_and_prune(void) {
    ht_adwin_pool_t ad_pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&ad_pool), HT_OK);

    ht_config_params_t params;
    ht_default_hat_reg_params(&params, 2);
    params.drift_window_threshold = 10U; /* Small threshold for test */
    params.switch_significance = (ht_float_t)0.05;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init_hat(&tree, &params, &ad_pool), HT_OK);

    /* 1. Simulate alternate tree instantiation */
    uint16_t alt_root = ht_node_alloc(&tree.memory);
    uint16_t alt_stats = ht_stats_alloc(&tree.memory);
    tree.memory.node_pool[alt_root].stats_idx = alt_stats;
    ht_var_init(&tree.memory.node_pool[alt_root].reg_stats);
    ht_node_drift_t *alt_nd = ht_tree_node_drift(&tree, alt_root);
    HT_ASSERT_NOT_NULL(alt_nd);
    alt_nd->adwin_idx = ht_adwin_pool_alloc(&ad_pool, NULL);

    ht_node_drift_t *root_nd = ht_tree_node_drift(&tree, tree.root_idx);
    HT_ASSERT_NOT_NULL(root_nd);
    root_nd->alt_tree_idx = alt_root;
    tree.n_alternate_trees++;

    /* 2. Configure alt tree with lower error than main tree (Switch case) */
    root_nd->mean_error = (ht_float_t)10.0;
    root_nd->error_m2 = (ht_float_t)5.0;
    root_nd->error_count = 50U;

    alt_nd->mean_error = (ht_float_t)1.0;
    alt_nd->error_m2 = (ht_float_t)2.0;
    alt_nd->error_count = 50U;

    ht_hat_action_t act = ht_hat_evaluate_reg_alternate(
        root_nd->mean_error,
        root_nd->error_m2 / (ht_float_t)49.0,
        root_nd->error_count,
        alt_nd->mean_error,
        alt_nd->error_m2 / (ht_float_t)49.0,
        alt_nd->error_count,
        tree.params.drift_window_threshold,
        tree.params.switch_significance
    );
    HT_ASSERT_EQ(act, HT_HAT_ACTION_SWITCH);

    /* 3. Configure alt tree with higher error (Prune case) */
    alt_nd->mean_error = (ht_float_t)25.0;
    act = ht_hat_evaluate_reg_alternate(
        root_nd->mean_error,
        root_nd->error_m2 / (ht_float_t)49.0,
        root_nd->error_count,
        alt_nd->mean_error,
        alt_nd->error_m2 / (ht_float_t)49.0,
        alt_nd->error_count,
        tree.params.drift_window_threshold,
        tree.params.switch_significance
    );
    HT_ASSERT_EQ(act, HT_HAT_ACTION_PRUNE);

    /* Clean up */
    HT_ASSERT_EQ(ht_tree_free_subtree(&tree, alt_root), HT_OK);
    HT_ASSERT_EQ(ht_tree_free_subtree(&tree, tree.root_idx), HT_OK);
}

static void test_tree_leaf_sgd_hybrid_normalization(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.leaf_prediction = HT_LEAF_MODEL;
    params.learning_rate = (ht_float_t)0.05;
    params.l2_regularization = (ht_float_t)1e-4;
    params.leaf_sgd_normalize = true;
    params.grace_period = 500U;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Features at vastly different scales:
     * x[0] in [0, 1]
     * x[1] in [0, 1000]
     * Target y = 5.0 * x[0] + 0.002 * x[1] + 3.0 */
    for (int i = 0; i < 400; ++i) {
        ht_float_t v0 = (ht_float_t)(i % 10) * (ht_float_t)0.1;
        ht_float_t v1 = (ht_float_t)((i * 37) % 1000);
        ht_float_t y = (ht_float_t)5.0 * v0 + (ht_float_t)0.002 * v1 + (ht_float_t)3.0;

        x[0].num = v0;
        x[1].num = v1;
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    /* Test prediction at unseen vector */
    x[0].num = (ht_float_t)0.5;
    x[1].num = (ht_float_t)500.0;
    /* Expected y = 5.0 * 0.5 + 0.002 * 500 + 3.0 = 2.5 + 1.0 + 3.0 = 6.5 */
    ht_float_t pred = ht_tree_predict_reg(&tree, x);
    HT_ASSERT_FLOAT_NEAR(pred, (ht_float_t)6.5, (ht_float_t)1.5);

    /* Verify weights are stable and bounded */
    uint16_t s_idx = tree.memory.node_pool[tree.root_idx].stats_idx;
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_weights[0] < (ht_float_t)100.0);
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_weights[0] > (ht_float_t)-100.0);
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_weights[1] < (ht_float_t)100.0);
    HT_ASSERT(tree.memory.stats_pool[s_idx].reg_weights[1] > (ht_float_t)-100.0);
}

static void test_tree_attached_prep_pipeline(void) {
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    params.grace_period = 50U;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    /* Set up a preprocessing pipeline with Hampel filter on feature 0 */
    ht_feature_type_t types[HT_MAX_FEATURES];
    memset(types, 0, sizeof(types));
    types[0] = HT_FEAT_NUMERIC;
    types[1] = HT_FEAT_NUMERIC;

    ht_prep_pipeline_t pipe;
    ht_prep_pipeline_init(&pipe, 2, types);
    ht_hampel_set_enabled(&pipe.hampel, 0, true);
    ht_hampel_set_threshold(&pipe.hampel, (ht_float_t)3.0);

    /* Attach pipeline to tree */
    HT_ASSERT_EQ(ht_tree_set_prep_pipeline(&tree, &pipe), HT_OK);
    HT_ASSERT(tree.prep_pipeline == &pipe);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    /* Train with cluster 0 around (1, 1) and cluster 1 around (10, 10),
     * injecting occasional sensor spikes (e.g. 100000.0) */
    for (int i = 0; i < 200; ++i) {
        uint8_t c = (i % 2 == 0) ? 0 : 1;
        ht_float_t base_v = (c == 0) ? (ht_float_t)1.0 : (ht_float_t)10.0;
        ht_float_t noise = (ht_float_t)((i % 5) - 2) * (ht_float_t)0.1;

        x[0].num = base_v + noise;
        x[1].num = base_v + noise;

        /* Inject spike every 25 samples */
        if (i > 10 && i % 25 == 0) {
            x[0].num = (ht_float_t)100000.0;
        }

        HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, c, (ht_float_t)1.0), HT_OK);
    }

    /* Test prediction: with spike injected at query time, Hampel replaces it with median */
    x[0].num = (ht_float_t)1.1;
    x[1].num = (ht_float_t)1.0;
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 0);

    x[0].num = (ht_float_t)10.2;
    x[1].num = (ht_float_t)9.9;
    HT_ASSERT_EQ(ht_tree_predict(&tree, x), 1);

    /* Detach pipeline */
    HT_ASSERT_EQ(ht_tree_set_prep_pipeline(&tree, NULL), HT_OK);
    HT_ASSERT(tree.prep_pipeline == NULL);
}

/* ========================================================================= */
/* --- Test Runner Entry Point --------------------------------------------- */
/* ========================================================================= */

void test_tree_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Arvore VFDT Completa (ht_tree)");

    /* Nível 1: Testes Unitários */
    HT_RUN_TEST(test_tree_default_params_and_init);
    HT_RUN_TEST(test_tree_node_promise_analytical);
    HT_RUN_TEST(test_tree_routing_numeric_and_categorical);
    HT_RUN_TEST(test_tree_routing_missing_nan_and_most_common_path);
    HT_RUN_TEST(test_tree_predict_majority_class);
    HT_RUN_TEST(test_tree_predict_naive_bayes_lse);
    HT_RUN_TEST(test_tree_predict_nba_adaptive_switching);

    /* Nível 2: Integração e Gerenciamento de Memória */
    HT_RUN_TEST(test_tree_learn_one_and_split);
    HT_RUN_TEST(test_tree_leaf_deactivation_and_reactivation);
    HT_RUN_TEST(test_tree_leaf_deactivation_least_promising_selection);
    HT_RUN_TEST(test_tree_leaf_deactivation_pool_exhaustion_stress);
    HT_RUN_TEST(test_tree_serialization_roundtrip_full_and_inference);

    /* Nível 1 & 2: Regressão Contínua (VFDT-R & HAT-R) */
    HT_RUN_TEST(test_tree_reg_default_params_and_init);
    HT_RUN_TEST(test_tree_reg_predict_leaf_mean);
    HT_RUN_TEST(test_tree_reg_predict_leaf_model_sgd_and_clipping);
    HT_RUN_TEST(test_tree_reg_predict_leaf_adaptive_fmse);
    HT_RUN_TEST(test_tree_reg_split_and_weight_inheritance);
    HT_RUN_TEST(test_tree_reg_leaf_deactivation_by_depth);
    HT_RUN_TEST(test_tree_reg_serialization_roundtrip);
    HT_RUN_TEST(test_tree_hat_reg_alternate_tree_switch_and_prune);

    /* Nível 2: Micro-Pipeline & Normalização SGD Híbrida */
    HT_RUN_TEST(test_tree_leaf_sgd_hybrid_normalization);
    HT_RUN_TEST(test_tree_attached_prep_pipeline);
}
