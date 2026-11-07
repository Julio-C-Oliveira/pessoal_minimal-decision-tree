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

void test_e2e_flow_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 3: Fluxo Ponta a Ponta Acumulado (E2E)");
    HT_RUN_TEST(test_e2e_sample_vector_ingestion);
    HT_RUN_TEST(test_e2e_math_numerical_pipeline);
    HT_RUN_TEST(test_e2e_memory_split_simulation);
    HT_RUN_TEST(test_e2e_stream_observer_pipeline);
}
