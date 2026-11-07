/**
 * @file test_e2e_flow.c
 * @brief Level 3 (Macro / E2E) Cumulative End-to-End Pipeline Tests.
 *
 * Exercises the complete data path available at the current milestone:
 * In Milestone 1.1, validates the end-to-end stream ingestion vector (ht_value_t[]),
 * missing value handling (NaN and negative categories), and numerical stability
 * across precision modes without any dynamic memory allocation.
 */

#include "test_harness.h"
#include "ht_memory.h"
#include "ht_observer.h"
#include "ht_criterion.h"
#include "ht_tree.h"
#include "ht_drift.h"

static void test_e2e_sample_vector_ingestion(void) {
    ht_value_t x[HT_MAX_FEATURES];
    ht_feature_type_t types[HT_MAX_FEATURES];

    /* Configure mixed schema: numeric (0..9), categorical (10..13), missing (14..15) */
    for (uint8_t i = 0; i < 10; ++i) {
        types[i] = HT_FEAT_NUMERIC;
        x[i].num = (ht_float_t)(i * 1.5 - 5.0);
    }
    for (uint8_t i = 10; i < 14; ++i) {
        types[i] = HT_FEAT_CATEGORICAL;
        x[i].cat = (int32_t)(i - 10);
    }

    /* Missing values */
    types[14] = HT_FEAT_NUMERIC;
    x[14].num = (ht_float_t)(0.0 / 0.0); /* NaN */

    types[15] = HT_FEAT_CATEGORICAL;
    x[15].cat = -1; /* Missing categorical */

    /* Process all features in stream sample */
    uint8_t valid_numeric_count = 0;
    uint8_t valid_categorical_count = 0;
    uint8_t missing_count = 0;

    for (uint8_t i = 0; i < HT_MAX_FEATURES; ++i) {
        if (types[i] == HT_FEAT_NUMERIC) {
            if (HT_ISNAN(x[i].num)) {
                missing_count++;
            } else {
                valid_numeric_count++;
                HT_ASSERT(HT_ISFINITE(x[i].num));
            }
        } else if (types[i] == HT_FEAT_CATEGORICAL) {
            if (x[i].cat < 0 || x[i].cat >= (int32_t)HT_MAX_CATEGORIES) {
                missing_count++;
            } else {
                valid_categorical_count++;
                HT_ASSERT(x[i].cat >= 0 && x[i].cat < (int32_t)HT_MAX_CATEGORIES);
            }
        }
    }

    HT_ASSERT_EQ(valid_numeric_count, 10);
    HT_ASSERT_EQ(valid_categorical_count, 4);
    HT_ASSERT_EQ(missing_count, 2);
}

static void test_e2e_math_numerical_pipeline(void) {
    /* Test normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2))) */
    ht_float_t z_vals[] = {
        (ht_float_t)-3.0,
        (ht_float_t)-1.96,
        (ht_float_t)0.0,
        (ht_float_t)1.96,
        (ht_float_t)3.0
    };
    ht_float_t prev_cdf = (ht_float_t)-1.0;

    for (size_t i = 0; i < sizeof(z_vals) / sizeof(z_vals[0]); ++i) {
        ht_float_t z = z_vals[i];
        ht_float_t cdf = (ht_float_t)0.5 * ((ht_float_t)1.0 + HT_ERF(z / HT_SQRT2));

        HT_ASSERT(HT_ISFINITE(cdf));
        HT_ASSERT(cdf >= (ht_float_t)0.0 && cdf <= (ht_float_t)1.0);
        HT_ASSERT(cdf > prev_cdf); /* Monotonically strictly increasing */
        prev_cdf = cdf;
    }

    /* Phi(0) must be exactly 0.5 */
    ht_float_t cdf_zero = (ht_float_t)0.5 * ((ht_float_t)1.0 + HT_ERF((ht_float_t)0.0 / HT_SQRT2));
    HT_ASSERT_FLOAT_NEAR(cdf_zero, (ht_float_t)0.5, 1e-6);

    /* Phi(1.96) is approximately 0.975002 */
    ht_float_t cdf_196 = (ht_float_t)0.5 * ((ht_float_t)1.0 + HT_ERF((ht_float_t)1.95996398454 / HT_SQRT2));
    HT_ASSERT_FLOAT_NEAR(cdf_196, (ht_float_t)0.975, 1e-3);

    /* Simulate Hoeffding bound: epsilon = sqrt((R^2 * ln(1/delta)) / (2n)) */
    ht_float_t r = HT_LOG2((ht_float_t)HT_MAX_CLASSES); /* log2(4) = 2.0 */
    ht_float_t delta = (ht_float_t)1e-7;
    ht_float_t ln_inv_delta = HT_LOG((ht_float_t)1.0 / delta);

    ht_float_t prev_eps = (ht_float_t)1e9;
    ht_float_t sample_counts[] = { (ht_float_t)200.0, (ht_float_t)1000.0, (ht_float_t)10000.0 };

    for (size_t i = 0; i < 3; ++i) {
        ht_float_t n = sample_counts[i];
        ht_float_t eps = HT_SQRT((r * r * ln_inv_delta) / ((ht_float_t)2.0 * n));
        HT_ASSERT(HT_ISFINITE(eps));
        HT_ASSERT(eps > (ht_float_t)0.0);
        HT_ASSERT(eps < prev_eps); /* Epsilon shrinks as n grows */
        prev_eps = eps;
    }
}

static ht_memory_pool_t s_e2e_pool;

static void test_e2e_memory_split_simulation(void) {
    ht_memory_init(&s_e2e_pool);

    /* 1. Allocate root node and its active leaf stats block */
    uint16_t root_idx = ht_node_alloc(&s_e2e_pool);
    HT_ASSERT_EQ(root_idx, 0);

    uint16_t root_stats = ht_stats_alloc(&s_e2e_pool);
    HT_ASSERT_EQ(root_stats, 0);

    ht_node_t *root = ht_node_get(&s_e2e_pool, root_idx);
    HT_ASSERT_NOT_NULL(root);
    root->stats_idx = root_stats;
    root->class_counts[0] = (ht_float_t)120.0;
    root->class_counts[1] = (ht_float_t)80.0;

    /* 2. Simulate split event on root */
    uint16_t mother_stats_idx = root->stats_idx;
    HT_ASSERT_NEQ(mother_stats_idx, HT_INVALID_IDX);

    /* Mark root as internal node and disconnect leaf stats */
    root->stats_idx = HT_INVALID_IDX;
    root->flags &= ~HT_NODE_IS_LEAF;
    root->feature_idx = 0;
    root->split_val.threshold = (ht_float_t)2.5;

    /* Return mother's stats_idx to stats_pool BEFORE allocating children's stats */
    ht_stats_free(&s_e2e_pool, mother_stats_idx);

    /* Allocate children nodes */
    uint16_t left_child = ht_node_alloc(&s_e2e_pool);
    uint16_t right_child = ht_node_alloc(&s_e2e_pool);
    HT_ASSERT_NEQ(left_child, HT_INVALID_IDX);
    HT_ASSERT_NEQ(right_child, HT_INVALID_IDX);

    root->left_child = left_child;
    root->right_child = right_child;

    /* Allocate leaf stats for children */
    uint16_t left_stats = ht_stats_alloc(&s_e2e_pool);
    uint16_t right_stats = ht_stats_alloc(&s_e2e_pool);

    /* O(1) recycling: left child immediately reuses the mother's returned stats index */
    HT_ASSERT_EQ(left_stats, mother_stats_idx);
    HT_ASSERT_NEQ(right_stats, left_stats);

    ht_node_get(&s_e2e_pool, left_child)->stats_idx = left_stats;
    ht_node_get(&s_e2e_pool, right_child)->stats_idx = right_stats;

    /* Verify current pool state: 3 nodes (1 internal, 2 active leaves) and 2 stats */
    HT_ASSERT_EQ(ht_node_count(&s_e2e_pool), 3);
    HT_ASSERT_EQ(ht_stats_count(&s_e2e_pool), 2);
    size_t bytes_used = ht_memory_bytes_used(&s_e2e_pool);
    HT_ASSERT(bytes_used <= sizeof(ht_memory_pool_t));
    HT_ASSERT(sizeof(ht_memory_pool_t) <= (1024 * 1024));

    /* 3. Simulate multi-level tree expansion with leaf queue */
    static uint16_t leaf_queue[256];
    uint16_t queue_head = 0;
    uint16_t queue_tail = 0;

    leaf_queue[queue_tail++] = left_child;
    leaf_queue[queue_tail++] = right_child;

    /* Split 30 more times (total 31 splits, 63 nodes, 32 active leaves) */
    for (uint32_t s = 0; s < 30; ++s) {
        uint16_t current_leaf = leaf_queue[queue_head++];
        ht_node_t *lnode = ht_node_get(&s_e2e_pool, current_leaf);
        HT_ASSERT_NOT_NULL(lnode);

        uint16_t old_st = lnode->stats_idx;
        HT_ASSERT_NEQ(old_st, HT_INVALID_IDX);

        /* Turn into internal node */
        lnode->stats_idx = HT_INVALID_IDX;
        lnode->flags &= ~HT_NODE_IS_LEAF;
        lnode->feature_idx = (uint8_t)(s % HT_MAX_FEATURES);

        /* Recycle mother stats */
        ht_stats_free(&s_e2e_pool, old_st);

        uint16_t lc = ht_node_alloc(&s_e2e_pool);
        uint16_t rc = ht_node_alloc(&s_e2e_pool);
        HT_ASSERT_NEQ(lc, HT_INVALID_IDX);
        HT_ASSERT_NEQ(rc, HT_INVALID_IDX);

        lnode->left_child = lc;
        lnode->right_child = rc;

        uint16_t ls = ht_stats_alloc(&s_e2e_pool);
        uint16_t rs = ht_stats_alloc(&s_e2e_pool);
        HT_ASSERT_NEQ(ls, HT_INVALID_IDX);
        HT_ASSERT_NEQ(rs, HT_INVALID_IDX);

        /* Mother's stats block was recycled */
        HT_ASSERT_EQ(ls, old_st);

        ht_node_get(&s_e2e_pool, lc)->stats_idx = ls;
        ht_node_get(&s_e2e_pool, rc)->stats_idx = rs;

        leaf_queue[queue_tail++] = lc;
        leaf_queue[queue_tail++] = rc;

        HT_ASSERT(ht_memory_bytes_used(&s_e2e_pool) <= (1024 * 1024));
    }

    /* 4. Complete teardown and memory cleanup without recursion */
    for (uint16_t i = 0; i < HT_MAX_NODES; ++i) {
        ht_node_t *n = ht_node_get(&s_e2e_pool, i);
        if (n && (n->flags & HT_NODE_IN_USE)) {
            if (n->stats_idx != HT_INVALID_IDX) {
                ht_stats_free(&s_e2e_pool, n->stats_idx);
                n->stats_idx = HT_INVALID_IDX;
            }
            ht_node_free(&s_e2e_pool, i);
        }
    }

    HT_ASSERT_EQ(ht_node_count(&s_e2e_pool), 0);
    HT_ASSERT_EQ(ht_stats_count(&s_e2e_pool), 0);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_e2e_pool), 0);
}

static void test_e2e_stream_observer_pipeline(void) {
    ht_memory_init(&s_e2e_pool);

    /* Allocate an active leaf node (node 0) and its stats block (stats 0) */
    uint16_t leaf_node_idx = ht_node_alloc(&s_e2e_pool);
    HT_ASSERT_EQ(leaf_node_idx, 0);

    uint16_t leaf_stats_idx = ht_stats_alloc(&s_e2e_pool);
    HT_ASSERT_EQ(leaf_stats_idx, 0);

    ht_node_t *leaf = ht_node_get(&s_e2e_pool, leaf_node_idx);
    HT_ASSERT_NOT_NULL(leaf);
    leaf->stats_idx = leaf_stats_idx;

    ht_leaf_stats_t *stats = ht_stats_get(&s_e2e_pool, leaf_stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    /* Mixed schema definition:
     * Features 0..11: Numeric (Gaussian observers)
     * Features 12..15: Categorical (Nominal observers)
     * Classes: 3 classes (0, 1, 2)
     */
    const uint8_t n_classes = 3;
    const uint8_t n_features = HT_MAX_FEATURES; /* 16 */
    const uint8_t n_numeric = 12;

    /* Synthetic stream generator: 600 samples (200 per class) with distinct distributions */
    for (uint32_t i = 0; i < 600; ++i) {
        uint8_t y = (uint8_t)(i % n_classes);
        ht_float_t weight = (ht_float_t)1.0;

        /* Increment leaf class counts */
        leaf->class_counts[y] += weight;

        /* Prepare feature vector */
        ht_value_t x[HT_MAX_FEATURES];

        /* Numeric features */
        for (uint8_t f = 0; f < n_numeric; ++f) {
            /* Class separation: class y has center at (y * 10.0 + f * 2.0) */
            ht_float_t base = (ht_float_t)(y * 10.0 + f * 2.0);
            /* Pseudo-random variation within [-3, 3] based on sample index */
            ht_float_t offset = (ht_float_t)((int32_t)((i * 37 + f * 13) % 7) - 3);

            /* Inject missing value (NaN) periodically on feature 11 to test robustness */
            if (f == 11 && (i % 25 == 0)) {
                x[f].num = (ht_float_t)(0.0 / 0.0);
            } else {
                x[f].num = base + offset;
            }

            /* Update Gaussian observer */
            ht_gaussian_update(stats->observers[f].num, y, x[f].num, weight);
        }

        /* Categorical features */
        for (uint8_t f = n_numeric; f < n_features; ++f) {
            /* Inject invalid/missing category periodically on feature 15 (29 is coprime with 3) */
            if (f == 15 && (i % 29 == 0)) {
                x[f].cat = -1;
            } else {
                /* Classes have preferred categories */
                x[f].cat = (int32_t)((y + (f - n_numeric) + (i % 2)) % HT_MAX_CATEGORIES);
            }

            /* Update Categorical observer */
            ht_categorical_update(&stats->observers[f].cat, y, x[f].cat, weight);
        }
    }

    /* 1. Verify class distribution in leaf */
    HT_ASSERT_FLOAT_EQ(leaf->class_counts[0], (ht_float_t)200.0);
    HT_ASSERT_FLOAT_EQ(leaf->class_counts[1], (ht_float_t)200.0);
    HT_ASSERT_FLOAT_EQ(leaf->class_counts[2], (ht_float_t)200.0);
    HT_ASSERT_FLOAT_EQ(leaf->class_counts[3], (ht_float_t)0.0);

    /* 2. Verify all numeric feature observers */
    for (uint8_t f = 0; f < n_numeric; ++f) {
        for (uint8_t c = 0; c < n_classes; ++c) {
            ht_gaussian_obs_t *g = &stats->observers[f].num[c];
            if (f == 11) {
                /* Samples with NaN were safely skipped */
                HT_ASSERT(g->count < (ht_float_t)200.0 && g->count >= (ht_float_t)190.0);
            } else {
                HT_ASSERT_FLOAT_EQ(g->count, (ht_float_t)200.0);
            }

            /* Verify mean and variance are strictly finite and sensible */
            HT_ASSERT(HT_ISFINITE(g->mean));
            ht_float_t var = ht_gaussian_variance(g);
            HT_ASSERT(HT_ISFINITE(var));
            HT_ASSERT(var > (ht_float_t)0.0);
            HT_ASSERT(g->min_val < g->max_val);

            /* Verify conditional density and log-density */
            ht_float_t p = ht_gaussian_cond_proba(g, g->mean);
            HT_ASSERT(p > (ht_float_t)0.0);
            ht_float_t log_p = ht_gaussian_cond_log_proba(g, g->mean);
            HT_ASSERT(HT_ISFINITE(log_p));
            HT_ASSERT_FLOAT_NEAR(HT_EXP(log_p), p, 1e-4);
        }

        /* 3. Evaluate split candidates for this numeric feature */
        ht_float_t suggested_splits[HT_GAUSSIAN_SPLITS];
        uint8_t n_splits = ht_gaussian_split_suggestions(stats->observers[f].num, n_classes, suggested_splits);
        HT_ASSERT(n_splits > 0);
        HT_ASSERT(n_splits <= HT_GAUSSIAN_SPLITS);

        /* Evaluate binary partition for each suggested split */
        for (uint8_t s = 0; s < n_splits; ++s) {
            ht_float_t split_threshold = suggested_splits[s];
            ht_float_t lhs[HT_MAX_CLASSES];
            ht_float_t rhs[HT_MAX_CLASSES];

            ht_gaussian_binary_split_dists(stats->observers[f].num, n_classes, split_threshold, lhs, rhs);

            for (uint8_t c = 0; c < n_classes; ++c) {
                ht_float_t expected_total = stats->observers[f].num[c].count;
                HT_ASSERT_FLOAT_NEAR(lhs[c] + rhs[c], expected_total, 1e-3);
                HT_ASSERT(lhs[c] >= (ht_float_t)0.0);
                HT_ASSERT(rhs[c] >= (ht_float_t)0.0);
            }
        }
    }

    /* 4. Verify all categorical feature observers */
    for (uint8_t f = n_numeric; f < n_features; ++f) {
        ht_categorical_obs_t *cat = &stats->observers[f].cat;

        for (uint8_t c = 0; c < n_classes; ++c) {
            ht_float_t class_cat_total = (ht_float_t)0.0;
            for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
                class_cat_total += cat->counts[c][v];
            }
            if (f == 15) {
                HT_ASSERT(class_cat_total <= (ht_float_t)200.0 && class_cat_total >= (ht_float_t)190.0);
            } else {
                HT_ASSERT_FLOAT_EQ(class_cat_total, (ht_float_t)200.0);
            }
        }

        /* Test binary split distributions for each category */
        for (int32_t cat_v = 0; cat_v < (int32_t)HT_MAX_CATEGORIES; ++cat_v) {
            if (ht_categorical_is_observed(cat, cat_v)) {
                ht_float_t lhs[HT_MAX_CLASSES];
                ht_float_t rhs[HT_MAX_CLASSES];

                ht_categorical_binary_split_dists(cat, n_classes, cat_v, lhs, rhs);

                for (uint8_t c = 0; c < n_classes; ++c) {
                    ht_float_t class_cat_total = (ht_float_t)0.0;
                    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
                        class_cat_total += cat->counts[c][v];
                    }
                    HT_ASSERT_FLOAT_EQ(lhs[c] + rhs[c], class_cat_total);
                }
            }
        }
    }

    /* 5. Assemble a candidate split record in stack memory (BranchFactory replacement) */
    ht_split_candidate_t cand;
    ht_split_candidate_init(&cand);
    cand.feature_idx = 0;
    cand.split_val.threshold = (ht_float_t)15.0;
    cand.merit = (ht_float_t)0.42;
    cand.is_categorical = 0;
    ht_gaussian_binary_split_dists(stats->observers[0].num, n_classes, cand.split_val.threshold,
                                   cand.lhs_dist, cand.rhs_dist);

    HT_ASSERT_EQ(cand.feature_idx, 0);
    HT_ASSERT_EQ(cand.is_categorical, 0);
    HT_ASSERT_FLOAT_EQ(cand.merit, (ht_float_t)0.42);

    /* 6. Clean up: recycle leaf stats back to pool */
    ht_stats_free(&s_e2e_pool, leaf_stats_idx);
    leaf->stats_idx = HT_INVALID_IDX;
    ht_node_free(&s_e2e_pool, leaf_node_idx);

    HT_ASSERT_EQ(ht_node_count(&s_e2e_pool), 0);
    HT_ASSERT_EQ(ht_stats_count(&s_e2e_pool), 0);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_e2e_pool), 0);
}

static void test_e2e_split_decision_pipeline(void) {
    /* 1. Initialize standalone memory pool */
    static ht_memory_pool_t pool;
    ht_memory_init(&pool);

    /* 2. Allocate root leaf node and associated stats */
    uint16_t root_idx = ht_node_alloc(&pool);
    HT_ASSERT_NEQ(root_idx, HT_INVALID_IDX);
    ht_node_t *root = &pool.node_pool[root_idx];

    uint16_t stats_idx = ht_stats_alloc(&pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);
    root->stats_idx = stats_idx;
    ht_leaf_stats_t *stats = &pool.stats_pool[stats_idx];

    /* 3. Configure mixed schema: 4 features
     * Feature 0: Numeric, discriminatory (class 0 in [1..4], class 1 in [6..9])
     * Feature 1: Numeric, noise (both classes uniformly spread in [0..10])
     * Feature 2: Categorical, discriminatory (class 0 has cat 0, class 1 has cat 1)
     * Feature 3: Categorical, noise (both classes in cat 0..3)
     */
    const uint8_t n_features = 4;
    const uint8_t n_classes = 2;
    ht_feature_type_t types[HT_MAX_FEATURES];
    types[0] = HT_FEAT_NUMERIC;
    types[1] = HT_FEAT_NUMERIC;
    types[2] = HT_FEAT_CATEGORICAL;
    types[3] = HT_FEAT_CATEGORICAL;

    /* 4. Stream 400 synthetic samples (200 of Class 0, 200 of Class 1) */
    for (int i = 0; i < 200; i++) {
        /* Sample for Class 0 */
        ht_value_t x0[HT_MAX_FEATURES];
        x0[0].num = (ht_float_t)(1.0 + (i % 30) * 0.1); /* [1.0 .. 3.9] */
        x0[1].num = (ht_float_t)((i % 50) * 0.2);        /* [0.0 .. 9.8] noise */
        x0[2].cat = 0;                                   /* Category 0 */
        x0[3].cat = (int32_t)(i % 4);                    /* Category noise */

        root->class_counts[0] += (ht_float_t)1.0;
        ht_gaussian_update(stats->observers[0].num, 0, x0[0].num, (ht_float_t)1.0);
        ht_gaussian_update(stats->observers[1].num, 0, x0[1].num, (ht_float_t)1.0);
        ht_categorical_update(&stats->observers[2].cat, 0, x0[2].cat, (ht_float_t)1.0);
        ht_categorical_update(&stats->observers[3].cat, 0, x0[3].cat, (ht_float_t)1.0);

        /* Sample for Class 1 */
        ht_value_t x1[HT_MAX_FEATURES];
        x1[0].num = (ht_float_t)(6.0 + (i % 30) * 0.1); /* [6.0 .. 8.9] */
        x1[1].num = (ht_float_t)((i % 50) * 0.2);        /* [0.0 .. 9.8] noise */
        x1[2].cat = 1;                                   /* Category 1 */
        x1[3].cat = (int32_t)(i % 4);                    /* Category noise */

        root->class_counts[1] += (ht_float_t)1.0;
        ht_gaussian_update(stats->observers[0].num, 1, x1[0].num, (ht_float_t)1.0);
        ht_gaussian_update(stats->observers[1].num, 1, x1[1].num, (ht_float_t)1.0);
        ht_categorical_update(&stats->observers[2].cat, 1, x1[2].cat, (ht_float_t)1.0);
        ht_categorical_update(&stats->observers[3].cat, 1, x1[3].cat, (ht_float_t)1.0);
    }

    HT_ASSERT_FLOAT_EQ(root->class_counts[0], (ht_float_t)200.0);
    HT_ASSERT_FLOAT_EQ(root->class_counts[1], (ht_float_t)200.0);
    ht_float_t total_weight = root->class_counts[0] + root->class_counts[1]; /* 400.0 */

    /* 5. Evaluate split using Gini criterion */
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);
    HT_ASSERT_NOT_NULL(crit);

    ht_split_candidate_t best, second_best;
    ht_find_best_split_candidates(stats, root->class_counts, types,
                                  n_features, n_classes, crit,
                                  (ht_float_t)0.01, true, &best, &second_best);

    /* 6. Verify top candidates: either Feature 0 or Feature 2 */
    HT_ASSERT_FLOAT_NEAR(best.merit, (ht_float_t)1.0, 1e-4);
    HT_ASSERT(best.feature_idx == 0 || best.feature_idx == 2);

    /* 7. Compute Hoeffding bound for delta = 1e-7 and total_weight = 400.0 */
    ht_float_t r = crit->range_of_merit(root->class_counts, n_classes);
    ht_float_t delta = (ht_float_t)1e-7;
    ht_float_t tau = (ht_float_t)0.05;
    ht_float_t eps = ht_hoeffding_bound(r, delta, total_weight);
    HT_ASSERT_FLOAT_NEAR(eps, (ht_float_t)0.141942, 1e-3);

    /* 8. Split decision with tau */
    HT_ASSERT(ht_should_split(&best, &second_best, (ht_float_t)0.0, tau));

    /* 9. Execute split: allocate left and right children in the pool */
    uint16_t left_child = ht_node_alloc(&pool);
    uint16_t right_child = ht_node_alloc(&pool);
    HT_ASSERT_NEQ(left_child, HT_INVALID_IDX);
    HT_ASSERT_NEQ(right_child, HT_INVALID_IDX);

    /* Structural conversion of root from leaf to internal branch */
    root->left_child = left_child;
    root->right_child = right_child;
    root->feature_idx = best.feature_idx;
    root->flags &= ~HT_NODE_IS_LEAF;
    if (best.is_categorical) {
        root->split_val.cat_val = best.split_val.cat_val;
        root->flags |= HT_NODE_SPLIT_CAT;
    } else {
        root->split_val.threshold = best.split_val.threshold;
    }

    /* Initialize child nodes with branch distributions */
    ht_node_t *left = &pool.node_pool[left_child];
    ht_node_t *right = &pool.node_pool[right_child];
    for (uint8_t c = 0; c < n_classes; c++) {
        left->class_counts[c] = best.lhs_dist[c];
        right->class_counts[c] = best.rhs_dist[c];
    }

    /* 10. IMMEDIATE STATS RECYCLING (Core VFDT rule): return leaf stats to pool */
    ht_stats_free(&pool, root->stats_idx);
    root->stats_idx = HT_INVALID_IDX;

    /* Verify memory usage: exactly 3 nodes allocated, 0 stats blocks allocated */
    HT_ASSERT_EQ(ht_node_count(&pool), 3);
    HT_ASSERT_EQ(ht_stats_count(&pool), 0);
    HT_ASSERT_EQ(ht_memory_bytes_used(&pool), 3 * sizeof(ht_node_t));

    /* 11. Cleanup all nodes */
    ht_node_free(&pool, left_child);
    ht_node_free(&pool, right_child);
    ht_node_free(&pool, root_idx);

    HT_ASSERT_EQ(ht_node_count(&pool), 0);
    HT_ASSERT_EQ(ht_memory_bytes_used(&pool), 0);
}

static void test_e2e_full_tree_prequential_and_serialization(void) {
    /* 1. Setup tree with 4 features (2 informative numeric, 1 categorical, 1 noise) */
    ht_tree_t tree;
    ht_config_params_t params;
    ht_default_params(&params, 4, 2);
    params.feature_types[0] = HT_FEAT_NUMERIC;
    params.feature_types[1] = HT_FEAT_NUMERIC;
    params.feature_types[2] = HT_FEAT_CATEGORICAL;
    params.feature_types[3] = HT_FEAT_NUMERIC;
    params.leaf_prediction = HT_LEAF_NBA;
    params.grace_period = 50;
    params.delta = (ht_float_t)1e-4;
    params.tau = (ht_float_t)0.05;

    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);
    HT_ASSERT(sizeof(ht_tree_t) <= 1024 * 1024);

    /* 2. Prequential stream of 1,000 samples */
    int early_correct = 0;
    int late_correct = 0;

    for (int i = 0; i < 1000; ++i) {
        ht_value_t x[HT_MAX_FEATURES];
        memset(x, 0, sizeof(x));

        /* Deterministic pseudo-random generation without dynamic allocation */
        ht_float_t r1 = (ht_float_t)((i * 37 + 13) % 100) / (ht_float_t)100.0;
        ht_float_t r2 = (ht_float_t)((i * 59 + 29) % 100) / (ht_float_t)100.0;

        uint8_t y;
        if (i % 2 == 0) {
            /* Class 0: x[0] in [1.0, 4.0], other features are overlapping / noise */
            x[0].num = (ht_float_t)1.0 + (ht_float_t)3.0 * r1;
            x[1].num = (ht_float_t)5.0 + (ht_float_t)3.0 * r2; /* overlapping */
            x[2].cat = (int32_t)(i % 3);                        /* independent categorical */
            x[3].num = r1 * (ht_float_t)10.0;                  /* noise */
            y = 0;
        } else {
            /* Class 1: x[0] in [6.0, 9.5], other features are overlapping / noise */
            x[0].num = (ht_float_t)6.0 + (ht_float_t)3.5 * r1;
            x[1].num = (ht_float_t)5.0 + (ht_float_t)3.0 * r2; /* overlapping */
            x[2].cat = (int32_t)(i % 3);                        /* independent categorical */
            x[3].num = r2 * (ht_float_t)10.0;                  /* noise */
            y = 1;
        }

        /* Test-then-train (prequential evaluation) */
        uint8_t pred = ht_tree_predict(&tree, x);
        if (pred == y) {
            if (i < 100) early_correct++;
            if (i >= 800) late_correct++;
        }

        HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    /* 3. Verify that learning and splitting took place */
    HT_ASSERT(tree.n_decision_nodes >= 1);
    HT_ASSERT(tree.n_active_leaves >= 2);
    HT_ASSERT(tree.max_depth_seen >= 1);
    HT_ASSERT(tree.train_weight_seen >= (ht_float_t)1000.0);

    /* 4. Verify accuracy improvement: late accuracy (> 90%) must significantly exceed early */
    double early_acc = (double)early_correct / 100.0;
    double late_acc = (double)late_correct / 200.0;
    HT_ASSERT(late_acc >= 0.90);
    HT_ASSERT(late_acc >= early_acc);

    /* 5. Memory budget check */
    HT_ASSERT(ht_memory_bytes_used(&tree.memory) <= 1024 * 1024);

    /* 6. Binary serialization: save and reload full model */
    size_t needed = 0;
    HT_ASSERT_EQ(ht_tree_save(&tree, NULL, 0, &needed, true), HT_OK);
    HT_ASSERT(needed < 65536);

    uint8_t save_buf[65536];
    size_t written = 0;
    HT_ASSERT_EQ(ht_tree_save(&tree, save_buf, sizeof(save_buf), &written, true), HT_OK);
    HT_ASSERT_EQ(written, needed);

    ht_tree_t loaded_tree;
    HT_ASSERT_EQ(ht_tree_load(&loaded_tree, save_buf, written), HT_OK);

    /* 7. Verify 100% bit-identical prediction match between original and loaded tree */
    for (int t = 0; t < 100; ++t) {
        ht_value_t x[HT_MAX_FEATURES];
        memset(x, 0, sizeof(x));
        x[0].num = (ht_float_t)(t * 0.1);
        x[1].num = (ht_float_t)(10.0 - t * 0.1);
        x[2].cat = t % 2;
        x[3].num = (ht_float_t)5.0;

        ht_float_t p_orig[HT_MAX_CLASSES];
        ht_float_t p_load[HT_MAX_CLASSES];

        HT_ASSERT_EQ(ht_tree_predict_proba(&tree, x, p_orig), HT_OK);
        HT_ASSERT_EQ(ht_tree_predict_proba(&loaded_tree, x, p_load), HT_OK);

        HT_ASSERT_FLOAT_NEAR(p_orig[0], p_load[0], 1e-6);
        HT_ASSERT_FLOAT_NEAR(p_orig[1], p_load[1], 1e-6);
        HT_ASSERT_EQ(ht_tree_predict(&tree, x), ht_tree_predict(&loaded_tree, x));
    }
}

static void test_e2e_drift_stream_pipeline(void) {
    ht_adwin_pool_t adwin_pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&adwin_pool), HT_OK);

    /* Allocate ADWIN detector from pool */
    uint16_t det_idx = ht_adwin_pool_alloc(&adwin_pool, NULL);
    HT_ASSERT_NEQ(det_idx, HT_INVALID_IDX);
    ht_adwin_t *ad = ht_adwin_pool_get(&adwin_pool, det_idx);
    HT_ASSERT_NOT_NULL(ad);

    /* Stream with 3 distinct regimes:
     * Regime 1: 0..499 -> mean ~ 0.0
     * Regime 2: 500..999 -> mean ~ 5.0 (drift 1)
     * Regime 3: 1000..1499 -> mean ~ -3.0 (drift 2)
     */
    uint32_t total_drifts_r2 = 0;
    uint32_t total_drifts_r3 = 0;
    uint32_t lcg = 987654321U;

    for (uint32_t t = 0; t < 1500; ++t) {
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t noise = ((ht_float_t)(lcg & 0xFFFFU) / (ht_float_t)65536.0 - (ht_float_t)0.5) * (ht_float_t)0.4;

        ht_float_t center;
        if (t < 500) {
            center = (ht_float_t)0.0;
        } else if (t < 1000) {
            center = (ht_float_t)5.0;
        } else {
            center = (ht_float_t)-3.0;
        }

        ht_float_t val = center + noise;
        bool drift = ht_adwin_update(ad, val);

        if (drift) {
            if (t >= 500 && t < 1000) {
                total_drifts_r2++;
            } else if (t >= 1000) {
                total_drifts_r3++;
            }
        }
    }

    /* Drift must be detected in regimes 2 and 3 */
    HT_ASSERT(total_drifts_r2 > 0);
    HT_ASSERT(total_drifts_r3 > 0);

    /* Final estimation must adapt close to -3.0 */
    HT_ASSERT_FLOAT_NEAR(ht_adwin_get_estimation(ad), (ht_float_t)-3.0, (ht_float_t)0.3);

    /* Combined memory budget check: ADWIN pool + Tree memory <= 1 MB */
    size_t adwin_bytes = ht_adwin_pool_bytes_used(&adwin_pool);
    HT_ASSERT(adwin_bytes <= (256U * 1024U));

    /* Free detector back to pool */
    HT_ASSERT_EQ(ht_adwin_pool_free(&adwin_pool, det_idx), HT_OK);
    HT_ASSERT_EQ(adwin_pool.allocated_count, 0U);
}

static void test_e2e_hat_abrupt_drift_recovery(void) {
    static ht_adwin_pool_t adwin_pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&adwin_pool), HT_OK);

    ht_config_params_t params;
    ht_default_hat_params(&params, 2, 2);
    params.grace_period = 50;
    params.delta = (ht_float_t)1e-3;
    params.drift_window_threshold = 50;
    params.switch_significance = (ht_float_t)0.05;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init_hat(&tree, &params, &adwin_pool), HT_OK);

    uint32_t lcg = 42;
    uint32_t correct_a = 0;
    uint32_t total_a = 0;

    /* Concept A: samples 0..999 (x0 > 0 -> class 1, else class 0) */
    for (uint32_t t = 0; t < 1000; ++t) {
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t x0 = ((ht_float_t)(lcg & 0xFFFFU) / (ht_float_t)32768.0) - (ht_float_t)1.0;
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t x1 = ((ht_float_t)(lcg & 0xFFFFU) / (ht_float_t)32768.0) - (ht_float_t)1.0;

        uint8_t y = (x0 > (ht_float_t)0.0) ? 1 : 0;
        ht_value_t x[HT_MAX_FEATURES];
        memset(x, 0, sizeof(x));
        x[0].num = x0;
        x[1].num = x1;

        if (t >= 500) {
            uint8_t pred = ht_tree_predict(&tree, x);
            if (pred == y) {
                correct_a++;
            }
            total_a++;
        }

        HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    /* Concept A prequential accuracy in late stage must be > 90% */
    ht_float_t acc_a = (ht_float_t)correct_a / (ht_float_t)total_a;
    HT_ASSERT(acc_a > (ht_float_t)0.90);
    HT_ASSERT(tree.memory.nodes_allocated > 1U); /* Tree has split */

    /* Concept B: samples 1000..2999 (abrupt concept inversion: x0 > 0 -> class 0, else 1) */
    uint32_t correct_b_late = 0;
    uint32_t total_b_late = 0;

    for (uint32_t t = 1000; t < 3000; ++t) {
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t x0 = ((ht_float_t)(lcg & 0xFFFFU) / (ht_float_t)32768.0) - (ht_float_t)1.0;
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t x1 = ((ht_float_t)(lcg & 0xFFFFU) / (ht_float_t)32768.0) - (ht_float_t)1.0;

        uint8_t y = (x0 > (ht_float_t)0.0) ? 0 : 1;
        ht_value_t x[HT_MAX_FEATURES];
        memset(x, 0, sizeof(x));
        x[0].num = x0;
        x[1].num = x1;

        if (t >= 2500) {
            uint8_t pred = ht_tree_predict(&tree, x);
            if (pred == y) {
                correct_b_late++;
            }
            total_b_late++;
        }

        HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    /* Concept B prequential accuracy in late stage must recover to > 85% */
    ht_float_t acc_b_late = (ht_float_t)correct_b_late / (ht_float_t)total_b_late;
    HT_ASSERT(acc_b_late > (ht_float_t)0.85);

    /* Assert alternate tree lifecycle events occurred */
    HT_ASSERT(tree.n_alternate_trees > 0U);
    HT_ASSERT(tree.n_switch_alternate_trees > 0U);

    /* Memory budget verification: tree memory + adwin pool <= 1 MB */
    size_t tree_bytes = ht_memory_bytes_used(&tree.memory);
    size_t adwin_bytes = ht_adwin_pool_bytes_used(&adwin_pool);
    HT_ASSERT((tree_bytes + adwin_bytes) <= (1024U * 1024U));

    /* Clean up entire tree and assert 0 leaks in all pools */
    HT_ASSERT_EQ(ht_tree_free_subtree(&tree, tree.root_idx), HT_OK);
    HT_ASSERT_EQ(tree.memory.nodes_allocated, 0U);
    HT_ASSERT_EQ(tree.memory.stats_allocated, 0U);
    HT_ASSERT_EQ(adwin_pool.allocated_count, 0U);
}

static void test_e2e_continuous_regression_stream_pipeline(void) {
    ht_memory_init(&s_e2e_pool);

    /* Allocate regression leaf node (node 0) and leaf stats block (stats 0) */
    uint16_t root_node_idx = ht_node_alloc(&s_e2e_pool);
    HT_ASSERT_EQ(root_node_idx, 0);

    uint16_t root_stats_idx = ht_stats_alloc(&s_e2e_pool);
    HT_ASSERT_EQ(root_stats_idx, 0);

    ht_node_t *root = ht_node_get(&s_e2e_pool, root_node_idx);
    HT_ASSERT_NOT_NULL(root);
    root->stats_idx = root_stats_idx;

    ht_leaf_stats_t *stats = ht_stats_get(&s_e2e_pool, root_stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    /* Initialize observers */
    for (uint8_t f = 0; f < 8; ++f) {
        ht_reg_numeric_init(&stats->observers[f].reg_num);
    }
    for (uint8_t f = 8; f < 12; ++f) {
        ht_reg_categorical_init(&stats->observers[f].reg_cat);
    }

    const uint32_t N_SAMPLES = 500U;
    uint32_t lcg = 777U;

    /* Continuous stream ingestion */
    for (uint32_t i = 0; i < N_SAMPLES; ++i) {
        ht_value_t x[HT_MAX_FEATURES];

        /* Features 0..7: Continuous numeric */
        for (uint8_t f = 0; f < 8; ++f) {
            lcg = lcg * 1664525U + 1013904223U;
            ht_float_t raw = (ht_float_t)(lcg & 0xFFFFU) / (ht_float_t)65536.0; /* [0.0 .. 1.0) */
            /* Feature 7 has occasional missing values (NaN) */
            if (f == 7 && (i % 25 == 0)) {
                x[f].num = (ht_float_t)(0.0 / 0.0);
            } else {
                x[f].num = (ht_float_t)(f * 5.0) + raw * (ht_float_t)10.0;
            }
        }

        /* Features 8..11: Nominal categorical */
        for (uint8_t f = 8; f < 12; ++f) {
            /* Feature 11 has occasional missing categories (-1) */
            if (f == 11 && (i % 31 == 0)) {
                x[f].cat = -1;
            } else {
                x[f].cat = (int32_t)((i + f) % HT_MAX_CATEGORIES);
            }
        }

        /* Features 12..15: Additional numeric features */
        for (uint8_t f = 12; f < HT_MAX_FEATURES; ++f) {
            lcg = lcg * 1664525U + 1013904223U;
            x[f].num = (ht_float_t)((lcg & 0xFFU) - 128);
        }

        /* Continuous regression target function:
         * y = 2.0 * x[0] - 1.5 * x[1] + 3.0 * (float)x[8] + noise
         */
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t noise = ((ht_float_t)(lcg & 0xFFU) / (ht_float_t)256.0 - (ht_float_t)0.5) * (ht_float_t)2.0;
        ht_float_t y = (ht_float_t)2.0 * x[0].num - (ht_float_t)1.5 * x[1].num +
                       (ht_float_t)3.0 * (ht_float_t)x[8].cat + noise;
        ht_float_t weight = (ht_float_t)1.0;

        /* Update leaf target variance statistics */
        ht_var_update(&root->reg_stats, y, weight);

        /* Update numeric regression observers */
        for (uint8_t f = 0; f < 8; ++f) {
            ht_reg_numeric_update(&stats->observers[f].reg_num, x[f].num, y, weight);
        }

        /* Update categorical regression observers */
        for (uint8_t f = 8; f < 12; ++f) {
            ht_reg_categorical_update(&stats->observers[f].reg_cat, x[f].cat, y, weight);
        }
    }

    /* 1. Verify target statistics in leaf */
    HT_ASSERT_FLOAT_EQ(root->reg_stats.count, (ht_float_t)N_SAMPLES);
    HT_ASSERT(HT_ISFINITE(root->reg_stats.mean));
    HT_ASSERT(root->reg_stats.m2 > (ht_float_t)0.0);
    ht_float_t target_var = ht_var_value(&root->reg_stats);
    HT_ASSERT(target_var > (ht_float_t)0.0);

    /* 2. Verify all numeric regression observers */
    for (uint8_t f = 0; f < 8; ++f) {
        ht_reg_numeric_obs_t *num_obs = &stats->observers[f].reg_num;

        /* All 5 slots must be populated and sorted by x_mean */
        for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
            HT_ASSERT(num_obs->slots[k].y_stats.count > (ht_float_t)0.0);
            HT_ASSERT(HT_ISFINITE(num_obs->slots[k].x_mean));
            HT_ASSERT(HT_ISFINITE(num_obs->slots[k].y_stats.mean));
            HT_ASSERT(num_obs->slots[k].y_stats.m2 >= (ht_float_t)0.0);
        }
        for (uint8_t k = 0; k < HT_REG_NUM_SLOTS - 1; ++k) {
            HT_ASSERT(num_obs->slots[k].x_mean <= num_obs->slots[k + 1].x_mean);
        }

        /* Check total weight conservation */
        ht_float_t slot_w = (ht_float_t)0.0;
        for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
            slot_w += num_obs->slots[k].y_stats.count;
        }
        if (f == 7) {
            /* NaN samples skipped */
            HT_ASSERT(slot_w < (ht_float_t)N_SAMPLES && slot_w >= (ht_float_t)(N_SAMPLES - 30));
        } else {
            HT_ASSERT_FLOAT_EQ(slot_w, (ht_float_t)N_SAMPLES);
        }
    }

    /* 3. Verify categorical regression observers */
    for (uint8_t f = 8; f < 12; ++f) {
        ht_reg_categorical_obs_t *cat_obs = &stats->observers[f].reg_cat;
        ht_float_t cat_w = (ht_float_t)0.0;
        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            cat_w += cat_obs->cat_stats[v].count;
            if (cat_obs->cat_stats[v].count > (ht_float_t)0.0) {
                HT_ASSERT(HT_ISFINITE(cat_obs->cat_stats[v].mean));
                HT_ASSERT(cat_obs->cat_stats[v].m2 >= (ht_float_t)0.0);
            }
        }
        if (f == 11) {
            HT_ASSERT(cat_w < (ht_float_t)N_SAMPLES && cat_w >= (ht_float_t)(N_SAMPLES - 30));
        } else {
            HT_ASSERT_FLOAT_EQ(cat_w, (ht_float_t)N_SAMPLES);
        }
    }

    /* 4. Memory budget verification */
    HT_ASSERT(ht_memory_bytes_used(&s_e2e_pool) <= (1024 * 1024));

    /* 5. Clean up: recycle leaf stats and node */
    ht_stats_free(&s_e2e_pool, root_stats_idx);
    root->stats_idx = HT_INVALID_IDX;
    ht_node_free(&s_e2e_pool, root_node_idx);

    HT_ASSERT_EQ(ht_node_count(&s_e2e_pool), 0);
    HT_ASSERT_EQ(ht_stats_count(&s_e2e_pool), 0);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_e2e_pool), 0);
}

void test_e2e_flow_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 3: Fluxo Ponta a Ponta Acumulado (E2E)");
    HT_RUN_TEST(test_e2e_sample_vector_ingestion);
    HT_RUN_TEST(test_e2e_math_numerical_pipeline);
    HT_RUN_TEST(test_e2e_memory_split_simulation);
    HT_RUN_TEST(test_e2e_stream_observer_pipeline);
    HT_RUN_TEST(test_e2e_split_decision_pipeline);
    HT_RUN_TEST(test_e2e_full_tree_prequential_and_serialization);
    HT_RUN_TEST(test_e2e_drift_stream_pipeline);
    HT_RUN_TEST(test_e2e_hat_abrupt_drift_recovery);
    HT_RUN_TEST(test_e2e_continuous_regression_stream_pipeline);
}
