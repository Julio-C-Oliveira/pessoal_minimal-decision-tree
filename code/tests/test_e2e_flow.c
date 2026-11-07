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

void test_e2e_flow_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 3: Fluxo Ponta a Ponta Acumulado (E2E)");
    HT_RUN_TEST(test_e2e_sample_vector_ingestion);
    HT_RUN_TEST(test_e2e_math_numerical_pipeline);
    HT_RUN_TEST(test_e2e_memory_split_simulation);
}
