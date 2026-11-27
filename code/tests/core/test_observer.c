/**
 * @file test_observer.c
 * @brief Level 1 & 2 Unit and Integration tests for Attribute Observers (ht_observer).
 */

#include "test_harness.h"
#include "ht_observer.h"
#include "ht_memory.h"

/* ========================================================================= */
/* --- Level 1: Gaussian Observer Unit Tests ------------------------------- */
/* ========================================================================= */

static void test_gaussian_init(void) {
    ht_gaussian_obs_t obs;
    obs.count = HT_CONST(10.0);
    obs.mean = HT_CONST(5.0);
    obs.m2 = HT_CONST(2.0);
    obs.min_val = HT_CONST(1.0);
    obs.max_val = HT_CONST(9.0);

    ht_gaussian_init(&obs);

    HT_ASSERT_FLOAT_EQ(obs.count, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(obs.mean, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(obs.m2, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(obs.min_val, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(obs.max_val, HT_CONST(0.0));
}

static void test_gaussian_welford_analytical(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    const ht_float_t samples[] = {
        HT_CONST(2.0), HT_CONST(4.0), HT_CONST(4.0), HT_CONST(4.0),
        HT_CONST(5.0), HT_CONST(5.0), HT_CONST(7.0), HT_CONST(9.0)
    };
    const uint8_t n_samples = sizeof(samples) / sizeof(samples[0]);

    for (uint8_t i = 0; i < n_samples; ++i) {
        ht_gaussian_update(obs, 0, samples[i], HT_CONST(1.0));
    }

    HT_ASSERT_FLOAT_EQ(obs[0].count, HT_CONST(8.0));
    HT_ASSERT_FLOAT_NEAR(obs[0].mean, HT_CONST(5.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs[0].m2, HT_CONST(32.0), HT_TEST_TOL(1e-3));
    HT_ASSERT_FLOAT_EQ(obs[0].min_val, HT_CONST(2.0));
    HT_ASSERT_FLOAT_EQ(obs[0].max_val, HT_CONST(9.0));

    ht_float_t expected_var = HT_CONST(32.0 / 7.0);
    ht_float_t calc_var = ht_gaussian_variance(&obs[0]);
    HT_ASSERT_FLOAT_NEAR(calc_var, expected_var, HT_TEST_TOL(1e-3));

    ht_float_t expected_std = HT_SQRT(expected_var);
    ht_float_t calc_std = ht_gaussian_stddev(&obs[0]);
    HT_ASSERT_FLOAT_NEAR(calc_std, expected_std, HT_TEST_TOL(1e-3));
}

static void test_gaussian_welford_weighted(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    ht_gaussian_update(obs, 1, HT_CONST(10.0), HT_CONST(2.0));
    ht_gaussian_update(obs, 1, HT_CONST(20.0), HT_CONST(2.0));

    HT_ASSERT_FLOAT_EQ(obs[1].count, HT_CONST(4.0));
    HT_ASSERT_FLOAT_NEAR(obs[1].mean, HT_CONST(15.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs[1].m2, HT_CONST(100.0), HT_TEST_TOL(1e-3));

    ht_float_t expected_var = HT_CONST(100.0 / 3.0);
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&obs[1]), expected_var, HT_TEST_TOL(1e-3));
}

static void test_gaussian_edge_cases(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    /* Edge Case 1: n = 0 */
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs[0]), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs[0]), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs[0], HT_CONST(5.0)), HT_CONST(0.0));
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs[0], HT_CONST(5.0)), -HT_INFINITY);

    /* Edge Case 2: n = 1 */
    ht_gaussian_update(obs, 0, HT_CONST(42.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(obs[0].count, HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(obs[0].mean, HT_CONST(42.0));
    HT_ASSERT_FLOAT_EQ(obs[0].m2, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs[0]), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs[0]), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs[0], HT_CONST(42.0)), HT_CONST(0.0));
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs[0], HT_CONST(42.0)), -HT_INFINITY);

    /* Edge Case 3: Constant feature (variance = 0, n = 3) */
    ht_gaussian_update(obs, 0, HT_CONST(42.0), HT_CONST(1.0));
    ht_gaussian_update(obs, 0, HT_CONST(42.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(obs[0].count, HT_CONST(3.0));
    HT_ASSERT_FLOAT_EQ(obs[0].m2, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs[0]), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs[0]), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs[0], HT_CONST(42.0)), HT_CONST(0.0));
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs[0], HT_CONST(42.0)), -HT_INFINITY);

    /* Edge Case 4: NaN input value must be ignored */
    ht_float_t nan_val = HT_NAN_VAL;
    ht_gaussian_update(obs, 0, nan_val, HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(obs[0].count, HT_CONST(3.0));

    /* Edge Case 5: Non-positive weight must be ignored */
    ht_gaussian_update(obs, 0, HT_CONST(100.0), HT_CONST(0.0));
    ht_gaussian_update(obs, 0, HT_CONST(100.0), HT_CONST(-2.0));
    HT_ASSERT_FLOAT_EQ(obs[0].count, HT_CONST(3.0));

    /* Edge Case 6: Out of bounds class_idx */
    ht_gaussian_update(obs, HT_MAX_CLASSES, HT_CONST(100.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(obs[0].count, HT_CONST(3.0));

    /* Edge Case 7: NULL observer pointer */
    ht_gaussian_update(NULL, 0, HT_CONST(10.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(NULL), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(NULL), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(NULL, HT_CONST(1.0)), HT_CONST(0.0));
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(NULL, HT_CONST(1.0)), -HT_INFINITY);
}

static void test_normal_cdf(void) {
    ht_float_t mean = HT_CONST(0.0);
    ht_float_t std = HT_CONST(1.0);

    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf(HT_CONST(0.0), mean, std), HT_CONST(0.5), HT_TEST_TOL(1e-6));
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf(HT_CONST(1.0), mean, std), HT_CONST(0.8413447), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf(HT_CONST(-1.0), mean, std), HT_CONST(0.1586553), HT_TEST_TOL(1e-5));

    ht_float_t cdf_pos = ht_normal_cdf(HT_CONST(1.5), mean, std);
    ht_float_t cdf_neg = ht_normal_cdf(HT_CONST(-1.5), mean, std);
    HT_ASSERT_FLOAT_NEAR(HT_ADD(cdf_pos, cdf_neg), HT_CONST(1.0), HT_TEST_TOL(1e-5));

    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf(HT_CONST(1.95996398454), mean, std), HT_CONST(0.975), HT_TEST_TOL(1e-4));

    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf(HT_CONST(10.0), mean, std), HT_CONST(1.0), HT_TEST_TOL(1e-6));
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf(HT_CONST(-10.0), mean, std), HT_CONST(0.0), HT_TEST_TOL(1e-6));

    HT_ASSERT_FLOAT_EQ(ht_normal_cdf(HT_CONST(4.0), HT_CONST(5.0), HT_CONST(0.0)), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_normal_cdf(HT_CONST(6.0), HT_CONST(5.0), HT_CONST(0.0)), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(ht_normal_cdf(HT_CONST(5.0), HT_CONST(5.0), HT_CONST(0.0)), HT_CONST(0.5));

    ht_float_t nan_val = HT_NAN_VAL;
    HT_ASSERT_NAN(ht_normal_cdf(nan_val, mean, std));
    HT_ASSERT_NAN(ht_normal_cdf(HT_CONST(0.0), nan_val, std));
}

static void test_gaussian_pdf_and_log_pdf(void) {
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

    obs.count = HT_CONST(100.0);
    obs.mean = HT_CONST(0.0);
    obs.m2 = HT_CONST(99.0);
    obs.min_val = HT_CONST(-3.0);
    obs.max_val = HT_CONST(3.0);

    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&obs), HT_CONST(1.0), HT_TEST_TOL(1e-6));

    ht_float_t expected_pdf_0 = HT_CONST(0.3989422804);
    ht_float_t calc_pdf_0 = ht_gaussian_cond_proba(&obs, HT_CONST(0.0));
    HT_ASSERT_FLOAT_NEAR(calc_pdf_0, expected_pdf_0, HT_TEST_TOL(1e-5));

    ht_float_t expected_log_pdf_0 = HT_CONST(-0.918938533);
    ht_float_t calc_log_pdf_0 = ht_gaussian_cond_log_proba(&obs, HT_CONST(0.0));
    HT_ASSERT_FLOAT_NEAR(calc_log_pdf_0, expected_log_pdf_0, HT_TEST_TOL(1e-5));

    HT_ASSERT_FLOAT_NEAR(HT_EXP(calc_log_pdf_0), calc_pdf_0, HT_TEST_TOL(1e-5));

    ht_float_t expected_pdf_1 = HT_CONST(0.2419707245);
    ht_float_t calc_pdf_1 = ht_gaussian_cond_proba(&obs, HT_CONST(1.0));
    HT_ASSERT_FLOAT_NEAR(calc_pdf_1, expected_pdf_1, HT_TEST_TOL(1e-5));

    ht_float_t expected_log_pdf_1 = HT_CONST(-1.418938533);
    ht_float_t calc_log_pdf_1 = ht_gaussian_cond_log_proba(&obs, HT_CONST(1.0));
    HT_ASSERT_FLOAT_NEAR(calc_log_pdf_1, expected_log_pdf_1, HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(HT_EXP(calc_log_pdf_1), calc_pdf_1, HT_TEST_TOL(1e-5));
}

static void test_gaussian_split_suggestions(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    ht_gaussian_update(obs, 0, HT_CONST(10.0), HT_CONST(1.0));
    ht_gaussian_update(obs, 0, HT_CONST(20.0), HT_CONST(1.0));
    ht_gaussian_update(obs, 1, HT_CONST(15.0), HT_CONST(1.0));
    ht_gaussian_update(obs, 1, HT_CONST(32.0), HT_CONST(1.0));

    ht_float_t splits[HT_GAUSSIAN_SPLITS];
    uint8_t n_splits = ht_gaussian_split_suggestions(obs, 2, splits);

    HT_ASSERT_EQ(n_splits, HT_GAUSSIAN_SPLITS);

    ht_float_t prev = HT_CONST(10.0);
    for (uint8_t i = 0; i < n_splits; ++i) {
        HT_ASSERT(splits[i] > prev);
        HT_ASSERT(splits[i] < HT_CONST(32.0));
        ht_float_t expected = HT_ADD(HT_CONST(10.0), HT_MUL(HT_CONST(2.0), HT_FROM_UINT((uint32_t)i + 1U)));
        HT_ASSERT_FLOAT_NEAR(splits[i], expected, HT_TEST_TOL(1e-4));
        prev = splits[i];
    }

    ht_gaussian_obs_t const_obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&const_obs[c]);
    }
    ht_gaussian_update(const_obs, 0, HT_CONST(5.0), HT_CONST(10.0));
    ht_gaussian_update(const_obs, 1, HT_CONST(5.0), HT_CONST(10.0));

    uint8_t n_const_splits = ht_gaussian_split_suggestions(const_obs, 2, splits);
    HT_ASSERT_EQ(n_const_splits, 0);

    ht_gaussian_obs_t empty_obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&empty_obs[c]);
    }
    HT_ASSERT_EQ(ht_gaussian_split_suggestions(empty_obs, 2, splits), 0);
}

static void test_gaussian_binary_split_dists(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    obs[0].count = HT_CONST(100.0);
    obs[0].mean = HT_CONST(15.0);
    obs[0].m2 = HT_CONST(99.0 * 4.0);
    obs[0].min_val = HT_CONST(10.0);
    obs[0].max_val = HT_CONST(20.0);

    obs[1].count = HT_CONST(50.0);
    obs[1].mean = HT_CONST(30.0);
    obs[1].m2 = HT_CONST(49.0 * 1.0);
    obs[1].min_val = HT_CONST(25.0);
    obs[1].max_val = HT_CONST(35.0);

    ht_float_t lhs[HT_MAX_CLASSES];
    ht_float_t rhs[HT_MAX_CLASSES];

    ht_gaussian_binary_split_dists(obs, 2, HT_CONST(5.0), lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(rhs[0], HT_CONST(100.0));
    HT_ASSERT_FLOAT_EQ(lhs[1], HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(rhs[1], HT_CONST(50.0));

    ht_gaussian_binary_split_dists(obs, 2, HT_CONST(40.0), lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], HT_CONST(100.0));
    HT_ASSERT_FLOAT_EQ(rhs[0], HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(lhs[1], HT_CONST(50.0));
    HT_ASSERT_FLOAT_EQ(rhs[1], HT_CONST(0.0));

    ht_gaussian_binary_split_dists(obs, 2, HT_CONST(15.0), lhs, rhs);
    HT_ASSERT_FLOAT_NEAR(lhs[0], HT_CONST(50.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(rhs[0], HT_CONST(50.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(HT_ADD(lhs[0], rhs[0]), HT_CONST(100.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_EQ(lhs[1], HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(rhs[1], HT_CONST(50.0));

    ht_gaussian_binary_split_dists(obs, 2, HT_CONST(22.0), lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], HT_CONST(100.0));
    HT_ASSERT_FLOAT_EQ(rhs[0], HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(lhs[1], HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(rhs[1], HT_CONST(50.0));
}

/* ========================================================================= */
/* --- Level 1: Categorical Nominal Observer Unit Tests ------------------- */
/* ========================================================================= */

static void test_categorical_basic_and_proba(void) {
    ht_categorical_obs_t cat;
    ht_categorical_init(&cat);

    ht_categorical_update(&cat, 0, 0, HT_CONST(3.0));
    ht_categorical_update(&cat, 0, 1, HT_CONST(1.0));
    ht_categorical_update(&cat, 1, 1, HT_CONST(2.0));
    ht_categorical_update(&cat, 1, 2, HT_CONST(8.0));

    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 0, 0), HT_CONST(0.75), HT_TEST_TOL(1e-6));
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 0, 1), HT_CONST(0.25), HT_TEST_TOL(1e-6));
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 0, 2), HT_CONST(0.0));

    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 1, 0), HT_CONST(0.0));
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 1, 1), HT_CONST(0.20), HT_TEST_TOL(1e-6));
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 1, 2), HT_CONST(0.80), HT_TEST_TOL(1e-6));

    HT_ASSERT(ht_categorical_is_observed(&cat, 0));
    HT_ASSERT(ht_categorical_is_observed(&cat, 1));
    HT_ASSERT(ht_categorical_is_observed(&cat, 2));
    HT_ASSERT(!ht_categorical_is_observed(&cat, 3));
    HT_ASSERT(!ht_categorical_is_observed(&cat, 4));
}

static void test_categorical_edge_cases(void) {
    ht_categorical_obs_t cat;
    ht_categorical_init(&cat);

    ht_categorical_update(&cat, 0, -1, HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 0, -1), HT_CONST(0.0));

    ht_categorical_update(&cat, 0, HT_MAX_CATEGORIES, HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 0, HT_MAX_CATEGORIES), HT_CONST(0.0));

    ht_categorical_update(&cat, HT_MAX_CLASSES, 0, HT_CONST(1.0));

    ht_categorical_update(&cat, 0, 0, HT_CONST(0.0));
    ht_categorical_update(&cat, 0, 0, HT_CONST(-5.0));

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            HT_ASSERT_FLOAT_EQ(cat.counts[c][v], HT_CONST(0.0));
        }
    }

    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 2, 0), HT_CONST(0.0));
    HT_ASSERT(!ht_categorical_is_observed(&cat, 0));

    ht_categorical_init(NULL);
    ht_categorical_update(NULL, 0, 0, HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(NULL, 0, 0), HT_CONST(0.0));
    HT_ASSERT(!ht_categorical_is_observed(NULL, 0));
}

static void test_categorical_binary_split_dists(void) {
    ht_categorical_obs_t cat;
    ht_categorical_init(&cat);

    ht_categorical_update(&cat, 0, 0, HT_CONST(3.0));
    ht_categorical_update(&cat, 0, 1, HT_CONST(1.0));
    ht_categorical_update(&cat, 1, 1, HT_CONST(2.0));
    ht_categorical_update(&cat, 1, 2, HT_CONST(8.0));

    ht_float_t lhs[HT_MAX_CLASSES];
    ht_float_t rhs[HT_MAX_CLASSES];

    ht_categorical_binary_split_dists(&cat, 2, 0, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], HT_CONST(3.0));
    HT_ASSERT_FLOAT_EQ(rhs[0], HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(lhs[1], HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(rhs[1], HT_CONST(10.0));

    ht_categorical_binary_split_dists(&cat, 2, 1, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(rhs[0], HT_CONST(3.0));
    HT_ASSERT_FLOAT_EQ(lhs[1], HT_CONST(2.0));
    HT_ASSERT_FLOAT_EQ(rhs[1], HT_CONST(8.0));

    for (uint8_t c = 0; c < 2; ++c) {
        HT_ASSERT_FLOAT_EQ(HT_ADD(lhs[c], rhs[c]), (c == 0) ? HT_CONST(4.0) : HT_CONST(10.0));
    }
}

static void test_split_candidate_struct(void) {
#ifndef HT_USE_DOUBLE
    HT_ASSERT_EQ(sizeof(ht_split_candidate_t), 44);
#else
    HT_ASSERT_EQ(sizeof(ht_split_candidate_t), 88);
#endif

    ht_split_candidate_t cand;
    cand.merit = HT_CONST(123.45);
    cand.feature_idx = 5;
    cand.is_categorical = 1;
    cand.split_val.threshold = HT_CONST(42.0);

    ht_split_candidate_init(&cand);

    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);
    HT_ASSERT_EQ(cand.feature_idx, HT_NO_FEATURE);
    HT_ASSERT_EQ(cand.is_categorical, 0);
    HT_ASSERT_FLOAT_EQ(cand.split_val.threshold, HT_CONST(0.0));

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        HT_ASSERT_FLOAT_EQ(cand.lhs_dist[c], HT_CONST(0.0));
        HT_ASSERT_FLOAT_EQ(cand.rhs_dist[c], HT_CONST(0.0));
    }
}

/* ========================================================================= */
/* --- Level 2: Meso Integration Tests (Observers in Pool Memory) ---------- */
/* ========================================================================= */

static ht_memory_pool_t s_observer_test_pool;

static void test_observers_in_leaf_stats_pool(void) {
    ht_memory_init(&s_observer_test_pool);

    uint16_t stats_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);

    ht_leaf_stats_t *stats = ht_stats_get(&s_observer_test_pool, stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].count, HT_CONST(0.0));
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].mean, HT_CONST(0.0));
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].m2, HT_CONST(0.0));
        }
    }

    ht_gaussian_update(stats->observers[0].num, 0, HT_CONST(10.0), HT_CONST(1.0));
    ht_gaussian_update(stats->observers[0].num, 0, HT_CONST(20.0), HT_CONST(1.0));
    ht_gaussian_update(stats->observers[0].num, 0, HT_CONST(30.0), HT_CONST(1.0));

    HT_ASSERT_FLOAT_EQ(stats->observers[0].num[0].count, HT_CONST(3.0));
    HT_ASSERT_FLOAT_NEAR(stats->observers[0].num[0].mean, HT_CONST(20.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&stats->observers[0].num[0]), HT_CONST(100.0), HT_TEST_TOL(1e-3));

    ht_categorical_update(&stats->observers[1].cat, 0, 2, HT_CONST(5.0));
    ht_categorical_update(&stats->observers[1].cat, 0, 4, HT_CONST(15.0));

    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&stats->observers[1].cat, 0, 2), HT_CONST(0.25), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&stats->observers[1].cat, 0, 4), HT_CONST(0.75), HT_TEST_TOL(1e-5));

    ht_stats_free(&s_observer_test_pool, stats_idx);
    HT_ASSERT_EQ(ht_stats_count(&s_observer_test_pool), 0);

    uint16_t recycled_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_EQ(recycled_idx, stats_idx);

    ht_leaf_stats_t *recycled_stats = ht_stats_get(&s_observer_test_pool, recycled_idx);
    HT_ASSERT_NOT_NULL(recycled_stats);
    HT_ASSERT_FLOAT_EQ(recycled_stats->observers[0].num[0].count, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(recycled_stats->observers[1].cat.counts[0][2], HT_CONST(0.0));
}

/* ========================================================================= */
/* --- Level 1 & 2: Incremental Variance & Regression Observers ------------ */
/* ========================================================================= */

static void test_var_stat_init_and_welford_analytical(void) {
    ht_var_stat_t v;
    v.count = HT_CONST(10.0);
    v.mean = HT_CONST(5.0);
    v.m2 = HT_VAR_M2_CONST(2.0);

    ht_var_init(&v);
    HT_ASSERT_FLOAT_EQ(v.count, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(v.mean, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(v.m2, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_var_value(&v), HT_CONST(0.0));

    ht_var_init(NULL);
    ht_var_update(NULL, HT_CONST(1.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(ht_var_value(NULL), HT_CONST(0.0));

    ht_var_update(&v, HT_CONST(42.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(v.count, HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(v.mean, HT_CONST(42.0));
    HT_ASSERT_FLOAT_EQ(v.m2, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_var_value(&v), HT_CONST(0.0));

    ht_var_update(&v, HT_CONST(42.0), HT_CONST(1.0));
    ht_var_update(&v, HT_CONST(42.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(v.count, HT_CONST(3.0));
    HT_ASSERT_FLOAT_EQ(v.m2, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_var_value(&v), HT_CONST(0.0));

    ht_float_t nan_val = HT_NAN_VAL;
    ht_var_update(&v, nan_val, HT_CONST(1.0));
    ht_var_update(&v, HT_CONST(10.0), HT_CONST(0.0));
    ht_var_update(&v, HT_CONST(10.0), HT_CONST(-1.0));
    HT_ASSERT_FLOAT_EQ(v.count, HT_CONST(3.0));

    ht_var_init(&v);
    const ht_float_t samples[] = {
        HT_CONST(2.0), HT_CONST(4.0), HT_CONST(4.0), HT_CONST(4.0),
        HT_CONST(5.0), HT_CONST(5.0), HT_CONST(7.0), HT_CONST(9.0)
    };
    for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        ht_var_update(&v, samples[i], HT_CONST(1.0));
    }
    HT_ASSERT_FLOAT_EQ(v.count, HT_CONST(8.0));
    HT_ASSERT_FLOAT_NEAR(v.mean, HT_CONST(5.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(v.m2, HT_VAR_M2_CONST(32.0), HT_TEST_TOL(1e-3));
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&v), HT_CONST(32.0 / 7.0), HT_TEST_TOL(1e-2));

    ht_var_stat_t wv;
    ht_var_init(&wv);
    ht_var_update(&wv, HT_CONST(10.0), HT_CONST(2.0));
    ht_var_update(&wv, HT_CONST(20.0), HT_CONST(2.0));
    HT_ASSERT_FLOAT_EQ(wv.count, HT_CONST(4.0));
    HT_ASSERT_FLOAT_NEAR(wv.mean, HT_CONST(15.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(wv.m2, HT_VAR_M2_CONST(100.0), HT_TEST_TOL(1e-3));
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&wv), HT_CONST(100.0 / 3.0), HT_TEST_TOL(1e-2));
}

static void test_var_stat_chan_combine_analytical(void) {
    ht_var_stat_t a;
    ht_var_init(&a);
    ht_var_update(&a, HT_CONST(10.0), HT_CONST(1.0));
    ht_var_update(&a, HT_CONST(20.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(a.count, HT_CONST(2.0));
    HT_ASSERT_FLOAT_NEAR(a.mean, HT_CONST(15.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(a.m2, HT_VAR_M2_CONST(50.0), HT_TEST_TOL(1e-3));

    ht_var_stat_t b;
    ht_var_init(&b);
    ht_var_update(&b, HT_CONST(30.0), HT_CONST(1.0));
    ht_var_update(&b, HT_CONST(40.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(b.count, HT_CONST(2.0));
    HT_ASSERT_FLOAT_NEAR(b.mean, HT_CONST(35.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(b.m2, HT_VAR_M2_CONST(50.0), HT_TEST_TOL(1e-3));

    ht_var_stat_t c = ht_var_combine(&a, &b);
    HT_ASSERT_FLOAT_EQ(c.count, HT_CONST(4.0));
    HT_ASSERT_FLOAT_NEAR(c.mean, HT_CONST(25.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(c.m2, HT_VAR_M2_CONST(500.0), HT_TEST_TOL(1e-3));
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&c), HT_CONST(500.0 / 3.0), HT_TEST_TOL(1e-2));

    ht_var_stat_t seq;
    ht_var_init(&seq);
    ht_var_update(&seq, HT_CONST(10.0), HT_CONST(1.0));
    ht_var_update(&seq, HT_CONST(20.0), HT_CONST(1.0));
    ht_var_update(&seq, HT_CONST(30.0), HT_CONST(1.0));
    ht_var_update(&seq, HT_CONST(40.0), HT_CONST(1.0));
    HT_ASSERT_FLOAT_EQ(c.count, seq.count);
    HT_ASSERT_FLOAT_NEAR(c.mean, seq.mean, HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(c.m2, seq.m2, HT_TEST_TOL(1e-3));

    ht_var_stat_t empty;
    ht_var_init(&empty);
    ht_var_stat_t c_empty_b = ht_var_combine(&a, &empty);
    HT_ASSERT_FLOAT_EQ(c_empty_b.count, a.count);
    HT_ASSERT_FLOAT_EQ(c_empty_b.mean, a.mean);
    HT_ASSERT_FLOAT_EQ(c_empty_b.m2, a.m2);

    ht_var_stat_t c_empty_a = ht_var_combine(&empty, &b);
    HT_ASSERT_FLOAT_EQ(c_empty_a.count, b.count);
    HT_ASSERT_FLOAT_EQ(c_empty_a.mean, b.mean);
    HT_ASSERT_FLOAT_EQ(c_empty_a.m2, b.m2);

    ht_var_stat_t c_null = ht_var_combine(NULL, NULL);
    HT_ASSERT_FLOAT_EQ(c_null.count, HT_CONST(0.0));
}

static void test_var_stat_chan_subtract_analytical(void) {
    ht_var_stat_t a, b;
    ht_var_init(&a);
    ht_var_init(&b);
    ht_var_update(&a, HT_CONST(10.0), HT_CONST(1.0));
    ht_var_update(&a, HT_CONST(20.0), HT_CONST(1.0));
    ht_var_update(&b, HT_CONST(30.0), HT_CONST(1.0));
    ht_var_update(&b, HT_CONST(40.0), HT_CONST(1.0));

    ht_var_stat_t c = ht_var_combine(&a, &b);

    ht_var_stat_t r = ht_var_subtract(&c, &a);
    HT_ASSERT_FLOAT_EQ(r.count, b.count);
    HT_ASSERT_FLOAT_NEAR(r.mean, b.mean, HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(r.m2, b.m2, HT_TEST_TOL(1e-3));
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&r), ht_var_value(&b), HT_TEST_TOL(1e-2));

    ht_var_stat_t r2 = ht_var_subtract(&c, &b);
    HT_ASSERT_FLOAT_EQ(r2.count, a.count);
    HT_ASSERT_FLOAT_NEAR(r2.mean, a.mean, HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(r2.m2, a.m2, HT_TEST_TOL(1e-3));

    ht_var_stat_t empty;
    ht_var_init(&empty);
    ht_var_stat_t r_empty = ht_var_subtract(&c, &empty);
    HT_ASSERT_FLOAT_EQ(r_empty.count, c.count);
    HT_ASSERT_FLOAT_NEAR(r_empty.mean, c.mean, HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(r_empty.m2, c.m2, HT_TEST_TOL(1e-3));

    ht_var_stat_t r_all = ht_var_subtract(&c, &c);
    HT_ASSERT_FLOAT_EQ(r_all.count, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(r_all.mean, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(r_all.m2, HT_CONST(0.0));

    ht_var_stat_t r_null = ht_var_subtract(NULL, &a);
    HT_ASSERT_FLOAT_EQ(r_null.count, HT_CONST(0.0));
}

static void test_reg_numeric_obs_ordered_slots_and_merging(void) {
    ht_reg_numeric_obs_t obs;
    ht_reg_numeric_init(&obs);

    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        HT_ASSERT_FLOAT_EQ(obs.slots[k].x_mean, HT_CONST(0.0));
        HT_ASSERT_FLOAT_EQ(obs.slots[k].y_stats.count, HT_CONST(0.0));
    }

    ht_reg_numeric_update(&obs, HT_CONST(10.0), HT_CONST(1.0), HT_CONST(1.0));
    ht_reg_numeric_update(&obs, HT_CONST(30.0), HT_CONST(3.0), HT_CONST(1.0));
    ht_reg_numeric_update(&obs, HT_CONST(20.0), HT_CONST(2.0), HT_CONST(1.0));
    ht_reg_numeric_update(&obs, HT_CONST(50.0), HT_CONST(5.0), HT_CONST(1.0));
    ht_reg_numeric_update(&obs, HT_CONST(40.0), HT_CONST(4.0), HT_CONST(1.0));

    HT_ASSERT_FLOAT_NEAR(obs.slots[0].x_mean, HT_CONST(10.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.slots[1].x_mean, HT_CONST(20.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.slots[2].x_mean, HT_CONST(30.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.slots[3].x_mean, HT_CONST(40.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.slots[4].x_mean, HT_CONST(50.0), HT_TEST_TOL(1e-5));
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        HT_ASSERT_FLOAT_EQ(obs.slots[k].y_stats.count, HT_CONST(1.0));
    }

    ht_reg_numeric_update(&obs, HT_CONST(21.0), HT_CONST(2.1), HT_CONST(1.0));

    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS - 1; ++k) {
        HT_ASSERT(obs.slots[k].x_mean <= obs.slots[k + 1].x_mean);
    }

    uint32_t lcg = 12345;
    ht_float_t sum_w = HT_CONST(0.0);
    ht_float_t sum_xw = HT_CONST(0.0);

    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        sum_w = HT_ADD(sum_w, obs.slots[k].y_stats.count);
        sum_xw = HT_ADD(sum_xw, HT_MUL(obs.slots[k].y_stats.count, obs.slots[k].x_mean));
    }

    for (int i = 0; i < 100; ++i) {
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t x_in = HT_MUL(HT_FROM_UINT(lcg % 1000U), HT_CONST(0.1));
        ht_float_t y_in = HT_SUB(HT_MUL(x_in, HT_CONST(2.5)), HT_CONST(10.0));
        ht_float_t w_in = HT_CONST(1.0);

        sum_w = HT_ADD(sum_w, w_in);
        sum_xw = HT_ADD(sum_xw, HT_MUL(x_in, w_in));

        ht_reg_numeric_update(&obs, x_in, y_in, w_in);

        for (uint8_t k = 0; k < HT_REG_NUM_SLOTS - 1; ++k) {
            HT_ASSERT(obs.slots[k].x_mean <= obs.slots[k + 1].x_mean);
            HT_ASSERT(obs.slots[k].y_stats.count > HT_CONST(0.0));
        }
    }

    ht_float_t total_slot_w = HT_CONST(0.0);
    ht_float_t total_slot_xw = HT_CONST(0.0);
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        total_slot_w = HT_ADD(total_slot_w, obs.slots[k].y_stats.count);
        total_slot_xw = HT_ADD(total_slot_xw, HT_MUL(obs.slots[k].y_stats.count, obs.slots[k].x_mean));
    }
    HT_ASSERT_FLOAT_NEAR(total_slot_w, sum_w, HT_TEST_TOL(1e-3));
    HT_ASSERT_FLOAT_NEAR(total_slot_xw, sum_xw, HT_TEST_TOL(0.5));

    ht_float_t nan_val = HT_NAN_VAL;
    ht_reg_numeric_update(&obs, nan_val, HT_CONST(1.0), HT_CONST(1.0));
    ht_reg_numeric_update(&obs, HT_CONST(1.0), nan_val, HT_CONST(1.0));
    ht_reg_numeric_update(&obs, HT_CONST(1.0), HT_CONST(1.0), HT_CONST(0.0));
    ht_reg_numeric_update(&obs, HT_CONST(1.0), HT_CONST(1.0), HT_CONST(-5.0));
    ht_reg_numeric_update(NULL, HT_CONST(1.0), HT_CONST(1.0), HT_CONST(1.0));

    ht_float_t after_edge_w = HT_CONST(0.0);
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        after_edge_w = HT_ADD(after_edge_w, obs.slots[k].y_stats.count);
    }
    HT_ASSERT_FLOAT_NEAR(after_edge_w, sum_w, HT_TEST_TOL(1e-3));
}

static void test_reg_categorical_obs_basic_and_edge_cases(void) {
    ht_reg_categorical_obs_t cat;
    ht_reg_categorical_init(&cat);

    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
        HT_ASSERT_FLOAT_EQ(cat.cat_stats[v].count, HT_CONST(0.0));
        HT_ASSERT_FLOAT_EQ(cat.cat_stats[v].mean, HT_CONST(0.0));
        HT_ASSERT_FLOAT_EQ(cat.cat_stats[v].m2, HT_CONST(0.0));
    }

    ht_reg_categorical_update(&cat, 0, HT_CONST(10.0), HT_CONST(1.0));
    ht_reg_categorical_update(&cat, 0, HT_CONST(20.0), HT_CONST(1.0));

    ht_reg_categorical_update(&cat, 2, HT_CONST(5.0), HT_CONST(1.0));
    ht_reg_categorical_update(&cat, 2, HT_CONST(5.0), HT_CONST(1.0));
    ht_reg_categorical_update(&cat, 2, HT_CONST(5.0), HT_CONST(1.0));

    HT_ASSERT_FLOAT_EQ(cat.cat_stats[0].count, HT_CONST(2.0));
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[0].mean, HT_CONST(15.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[0].m2, HT_VAR_M2_CONST(50.0), HT_TEST_TOL(1e-3));
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&cat.cat_stats[0]), HT_CONST(50.0), HT_TEST_TOL(1e-2));

    HT_ASSERT_FLOAT_EQ(cat.cat_stats[2].count, HT_CONST(3.0));
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[2].mean, HT_CONST(5.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[2].m2, HT_CONST(0.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_EQ(ht_var_value(&cat.cat_stats[2]), HT_CONST(0.0));

    HT_ASSERT_FLOAT_EQ(cat.cat_stats[1].count, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(cat.cat_stats[3].count, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(cat.cat_stats[4].count, HT_CONST(0.0));

    ht_reg_categorical_update(&cat, -1, HT_CONST(10.0), HT_CONST(1.0));
    ht_reg_categorical_update(&cat, HT_MAX_CATEGORIES, HT_CONST(10.0), HT_CONST(1.0));
    ht_float_t nan_val = HT_NAN_VAL;
    ht_reg_categorical_update(&cat, 0, nan_val, HT_CONST(1.0));
    ht_reg_categorical_update(&cat, 0, HT_CONST(10.0), HT_CONST(0.0));
    ht_reg_categorical_update(&cat, 0, HT_CONST(10.0), HT_CONST(-2.0));
    ht_reg_categorical_update(NULL, 0, HT_CONST(10.0), HT_CONST(1.0));
    ht_reg_categorical_init(NULL);

    HT_ASSERT_FLOAT_EQ(cat.cat_stats[0].count, HT_CONST(2.0));
}

static void test_reg_observers_in_leaf_stats_pool(void) {
    ht_memory_init(&s_observer_test_pool);

    uint16_t stats_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);

    ht_leaf_stats_t *stats = ht_stats_get(&s_observer_test_pool, stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    ht_reg_numeric_update(&stats->observers[0].reg_num, HT_CONST(5.0), HT_CONST(12.0), HT_CONST(1.0));
    ht_reg_numeric_update(&stats->observers[0].reg_num, HT_CONST(15.0), HT_CONST(24.0), HT_CONST(1.0));

    HT_ASSERT_FLOAT_NEAR(stats->observers[0].reg_num.slots[0].x_mean, HT_CONST(5.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(stats->observers[0].reg_num.slots[1].x_mean, HT_CONST(15.0), HT_TEST_TOL(1e-5));

    ht_reg_categorical_update(&stats->observers[1].reg_cat, 3, HT_CONST(42.0), HT_CONST(2.0));
    HT_ASSERT_FLOAT_EQ(stats->observers[1].reg_cat.cat_stats[3].count, HT_CONST(2.0));
    HT_ASSERT_FLOAT_NEAR(stats->observers[1].reg_cat.cat_stats[3].mean, HT_CONST(42.0), HT_TEST_TOL(1e-5));

    ht_stats_free(&s_observer_test_pool, stats_idx);
    HT_ASSERT_EQ(ht_stats_count(&s_observer_test_pool), 0);

    uint16_t realloc_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_EQ(realloc_idx, stats_idx);

    ht_leaf_stats_t *re_stats = ht_stats_get(&s_observer_test_pool, realloc_idx);
    HT_ASSERT_NOT_NULL(re_stats);
    HT_ASSERT_FLOAT_EQ(re_stats->observers[0].reg_num.slots[0].y_stats.count, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(re_stats->observers[1].reg_cat.cat_stats[3].count, HT_CONST(0.0));

    ht_stats_free(&s_observer_test_pool, realloc_idx);
}

/* ========================================================================= */
/* --- Level 1 & 2: Histogram Non-Parametric Observer Tests ---------------- */
/* ========================================================================= */

static void test_hist_obs_basic_and_merging(void) {
    ht_hist_obs_t obs;
    ht_hist_init(&obs);

    ht_hist_update(&obs, 0, HT_CONST(10.0), HT_CONST(1.0));
    ht_hist_update(&obs, 0, HT_CONST(20.0), HT_CONST(1.0));
    ht_hist_update(&obs, 0, HT_CONST(30.0), HT_CONST(1.0));
    ht_hist_update(&obs, 0, HT_CONST(40.0), HT_CONST(1.0));

    HT_ASSERT_FLOAT_NEAR(obs.bins[0].center, HT_CONST(10.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.bins[1].center, HT_CONST(20.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.bins[2].center, HT_CONST(30.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.bins[3].center, HT_CONST(40.0), HT_TEST_TOL(1e-5));

    ht_hist_update(&obs, 0, HT_CONST(11.0), HT_CONST(1.0));

    HT_ASSERT_FLOAT_NEAR(obs.bins[0].center, HT_CONST(10.5), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(obs.bins[0].counts[0], HT_CONST(2.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(obs.bins[1].center, HT_CONST(20.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(obs.bins[2].center, HT_CONST(30.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(obs.bins[3].center, HT_CONST(40.0), HT_TEST_TOL(1e-4));

    ht_float_t total = HT_CONST(0.0);
    for (uint8_t i = 0; i < HT_HIST_MAX_BINS; ++i) {
        total = HT_ADD(total, obs.bins[i].counts[0]);
    }
    HT_ASSERT_FLOAT_NEAR(total, HT_CONST(5.0), HT_TEST_TOL(1e-5));
}

static void test_hist_obs_split_dists_and_proba(void) {
    ht_hist_obs_t obs;
    ht_hist_init(&obs);

    ht_hist_update(&obs, 0, HT_CONST(1.0), HT_CONST(1.0));
    ht_hist_update(&obs, 0, HT_CONST(2.0), HT_CONST(1.0));
    ht_hist_update(&obs, 1, HT_CONST(10.0), HT_CONST(1.0));
    ht_hist_update(&obs, 1, HT_CONST(11.0), HT_CONST(1.0));

    ht_float_t splits[HT_GAUSSIAN_SPLITS];
    uint8_t n_splits = ht_hist_split_suggestions(&obs, 2, splits);
    HT_ASSERT(n_splits >= 1);

    ht_float_t lhs[HT_MAX_CLASSES];
    ht_float_t rhs[HT_MAX_CLASSES];
    ht_hist_binary_split_dists(&obs, 2, HT_CONST(5.0), lhs, rhs);

    HT_ASSERT_FLOAT_NEAR(lhs[0], HT_CONST(2.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(lhs[1], HT_CONST(0.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(rhs[0], HT_CONST(0.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(rhs[1], HT_CONST(2.0), HT_TEST_TOL(1e-4));

    ht_float_t p0 = ht_hist_cond_proba(&obs, 0, HT_CONST(1.5));
    ht_float_t p1 = ht_hist_cond_proba(&obs, 1, HT_CONST(1.5));
    HT_ASSERT(p0 > HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(p1, HT_CONST(0.0));
}

/* ========================================================================= */
/* --- Sufficient Statistics Merge Tests (Federated / Parallel) ------------ */
/* ========================================================================= */

static void test_observer_gaussian_and_categorical_merge_exact(void) {
    ht_gaussian_obs_t single[HT_MAX_CLASSES];
    ht_gaussian_obs_t part_a[HT_MAX_CLASSES];
    ht_gaussian_obs_t part_b[HT_MAX_CLASSES];
    memset(single, 0, sizeof(single));
    memset(part_a, 0, sizeof(part_a));
    memset(part_b, 0, sizeof(part_b));

    ht_categorical_obs_t cat_single, cat_a, cat_b;
    ht_categorical_init(&cat_single);
    ht_categorical_init(&cat_a);
    ht_categorical_init(&cat_b);

    for (int i = 0; i < 60; ++i) {
        uint8_t c = (uint8_t)(i % 3);
        ht_float_t v = HT_ADD(HT_SUB(HT_MUL(HT_FROM_INT(i), HT_CONST(0.75)), HT_CONST(10.0)),
                              HT_MUL(HT_FROM_UINT(c), HT_CONST(5.0)));
        int32_t cat = i % HT_MAX_CATEGORIES;
        ht_float_t w = HT_ADD(HT_CONST(1.0), HT_MUL(HT_FROM_INT(i % 2), HT_CONST(0.5)));

        ht_gaussian_update(single, c, v, w);
        ht_categorical_update(&cat_single, c, cat, w);

        if (i < 25) {
            ht_gaussian_update(part_a, c, v, w);
            ht_categorical_update(&cat_a, c, cat, w);
        } else {
            ht_gaussian_update(part_b, c, v, w);
            ht_categorical_update(&cat_b, c, cat, w);
        }
    }

    for (uint8_t c = 0; c < 3; ++c) {
        ht_gaussian_merge(&part_a[c], &part_b[c]);
        HT_ASSERT_FLOAT_NEAR(part_a[c].count, single[c].count, HT_TEST_TOL(1e-4));
        HT_ASSERT_FLOAT_NEAR(part_a[c].mean, single[c].mean, HT_TEST_TOL(1e-3));
        HT_ASSERT_FLOAT_NEAR(part_a[c].m2, single[c].m2, HT_TEST_TOL(0.1));
        HT_ASSERT_FLOAT_EQ(part_a[c].min_val, single[c].min_val);
        HT_ASSERT_FLOAT_EQ(part_a[c].max_val, single[c].max_val);
    }

    ht_categorical_merge(&cat_a, &cat_b, 3);
    for (uint8_t c = 0; c < 3; ++c) {
        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            HT_ASSERT_FLOAT_EQ(cat_a.counts[c][v], cat_single.counts[c][v]);
        }
    }
}

static void test_observer_hist_and_reg_merge(void) {
    ht_hist_obs_t h_a, h_b;
    ht_hist_init(&h_a);
    ht_hist_init(&h_b);

    for (int i = 0; i < 20; ++i) {
        ht_hist_update(&h_a, (uint8_t)(i % 2), HT_MUL(HT_FROM_INT(i), HT_CONST(0.5)), HT_CONST(1.0));
        ht_hist_update(&h_b, (uint8_t)(i % 2), HT_ADD(HT_CONST(10.0), HT_MUL(HT_FROM_INT(i), HT_CONST(0.5))), HT_CONST(1.0));
    }
    ht_hist_merge(&h_a, &h_b, 2);

    ht_float_t total_w = HT_CONST(0.0);
    for (uint8_t b = 0; b < HT_HIST_MAX_BINS; ++b) {
        if (b > 0) {
            HT_ASSERT(h_a.bins[b].center >= h_a.bins[b - 1].center);
        }
        total_w = HT_ADD(total_w, HT_ADD(h_a.bins[b].counts[0], h_a.bins[b].counts[1]));
    }
    HT_ASSERT_FLOAT_NEAR(total_w, HT_CONST(40.0), HT_TEST_TOL(1e-4));

    ht_reg_numeric_obs_t rn_a, rn_b;
    ht_reg_numeric_init(&rn_a);
    ht_reg_numeric_init(&rn_b);
    ht_reg_categorical_obs_t rc_a, rc_b, rc_single;
    ht_reg_categorical_init(&rc_a);
    ht_reg_categorical_init(&rc_b);
    ht_reg_categorical_init(&rc_single);

    for (int i = 0; i < 30; ++i) {
        ht_float_t xv = HT_FROM_INT(i);
        ht_float_t yv = HT_ADD(HT_MUL(HT_CONST(2.0), xv), HT_CONST(1.0));
        int32_t cat = i % HT_MAX_CATEGORIES;
        ht_reg_categorical_update(&rc_single, cat, yv, HT_CONST(1.0));
        if (i < 15) {
            ht_reg_numeric_update(&rn_a, xv, yv, HT_CONST(1.0));
            ht_reg_categorical_update(&rc_a, cat, yv, HT_CONST(1.0));
        } else {
            ht_reg_numeric_update(&rn_b, xv, yv, HT_CONST(1.0));
            ht_reg_categorical_update(&rc_b, cat, yv, HT_CONST(1.0));
        }
    }

    ht_reg_numeric_merge(&rn_a, &rn_b);
    ht_float_t reg_n = HT_CONST(0.0);
    for (uint8_t s = 0; s < HT_REG_NUM_SLOTS; ++s) {
        if (s > 0) {
            HT_ASSERT(rn_a.slots[s].x_mean >= rn_a.slots[s - 1].x_mean);
        }
        reg_n = HT_ADD(reg_n, rn_a.slots[s].y_stats.count);
    }
    HT_ASSERT_FLOAT_NEAR(reg_n, HT_CONST(30.0), HT_TEST_TOL(1e-4));

    ht_reg_categorical_merge(&rc_a, &rc_b);
    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
        HT_ASSERT_FLOAT_NEAR(rc_a.cat_stats[v].count, rc_single.cat_stats[v].count, HT_TEST_TOL(1e-4));
        HT_ASSERT_FLOAT_NEAR(rc_a.cat_stats[v].mean, rc_single.cat_stats[v].mean, HT_TEST_TOL(1e-3));
        HT_ASSERT_FLOAT_NEAR(rc_a.cat_stats[v].m2, rc_single.cat_stats[v].m2, HT_TEST_TOL(0.1));
    }
}

/* ========================================================================= */
/* --- Level 2: Long-Stream Block-Rescaling & High-Amplitude Q24.8 Tests --- */
/* ========================================================================= */

static void test_observer_100k_stream_block_rescaling(void) {
    ht_gaussian_obs_t g_obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&g_obs[c]);
    }
    ht_categorical_obs_t c_obs;
    ht_categorical_init(&c_obs);
    ht_var_stat_t v_obs;
    ht_var_init(&v_obs);

    /* Stream 100,000 samples (well beyond Q16.16 max 32767):
     * Class 0: alternating 8.0 and 12.0 -> mean = 10.0, var = 4.0
     * Class 1: alternating 18.0 and 22.0 -> mean = 20.0, var = 4.0
     * Categorical: cat 0 (75%), cat 1 (25%) for class 0
     */
    for (uint32_t i = 0; i < 100000U; ++i) {
        ht_float_t x0 = ((i & 1U) == 0U) ? HT_CONST(8.0) : HT_CONST(12.0);
        ht_float_t x1 = ((i & 1U) == 0U) ? HT_CONST(18.0) : HT_CONST(22.0);
        ht_gaussian_update(g_obs, 0, x0, HT_CONST(1.0));
        ht_gaussian_update(g_obs, 1, x1, HT_CONST(1.0));
        ht_categorical_update(&c_obs, 0, ((i & 3U) == 0U) ? 1 : 0, HT_CONST(1.0));
        ht_var_update(&v_obs, x0, HT_CONST(1.0));
    }

    /* Zero overflow / NaN / saturation, and moments preserved */
    HT_ASSERT(HT_ISFINITE(g_obs[0].mean));
    HT_ASSERT(HT_ISFINITE(g_obs[0].m2));
    HT_ASSERT_FLOAT_NEAR(g_obs[0].mean, HT_CONST(10.0), HT_TEST_TOL(0.1));
    HT_ASSERT_FLOAT_NEAR(g_obs[1].mean, HT_CONST(20.0), HT_TEST_TOL(0.1));
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&g_obs[0]), HT_CONST(4.0), HT_TEST_TOL(0.25));
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&g_obs[1]), HT_CONST(4.0), HT_TEST_TOL(0.25));

    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&c_obs, 0, 0), HT_CONST(0.75), HT_TEST_TOL(0.02));
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&c_obs, 0, 1), HT_CONST(0.25), HT_TEST_TOL(0.02));

    HT_ASSERT(HT_ISFINITE(v_obs.mean));
    HT_ASSERT_FLOAT_NEAR(v_obs.mean, HT_CONST(10.0), HT_TEST_TOL(0.1));
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&v_obs), HT_CONST(4.0), HT_TEST_TOL(0.25));
}

static void test_var_stat_high_amplitude_regression_q24_8(void) {
    /* Mimics NYC Taxi / Bikes target range: y in [100, 4800] */
    ht_var_stat_t v;
    ht_var_init(&v);
    for (int i = 0; i < 200; ++i) {
        ht_float_t y = ((i & 1) == 0) ? HT_CONST(200.0) : HT_CONST(1200.0);
        ht_var_update(&v, y, HT_CONST(1.0));
    }
    /* True mean = 700.0, true population variance = 500^2 = 250,000.
     * Under Q26.6 Block-Rescaling, effective window is ~67 samples so last alternating
     * sample shifts mean by <= ~8.0 (~1%) while keeping variance within 2% without overflow. */
    HT_ASSERT(HT_ISFINITE(v.mean));
    HT_ASSERT_FLOAT_NEAR(v.mean, HT_CONST(700.0), HT_TEST_TOL(15.0));
#ifdef HT_USE_FIXED_POINT
    int64_t var_q8 = ht_var_value_q8(&v);
    int64_t expected_q8 = 251256LL << HT_FIXED_M2_SHIFT;
    int64_t diff = (var_q8 > expected_q8) ? (var_q8 - expected_q8) : (expected_q8 - var_q8);
    HT_ASSERT(diff < (10000LL << HT_FIXED_M2_SHIFT));
#else
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&v), HT_CONST(251256.28), HT_TEST_TOL(5.0));
#endif
}

/* ========================================================================= */
/* --- Test Suite Runner --------------------------------------------------- */
/* ========================================================================= */

void test_observer_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Observadores de Atributos (ht_observer)");

    /* Nível 1: Gaussian Welford & CDF */
    HT_RUN_TEST(test_gaussian_init);
    HT_RUN_TEST(test_gaussian_welford_analytical);
    HT_RUN_TEST(test_gaussian_welford_weighted);
    HT_RUN_TEST(test_gaussian_edge_cases);
    HT_RUN_TEST(test_normal_cdf);
    HT_RUN_TEST(test_gaussian_pdf_and_log_pdf);
    HT_RUN_TEST(test_gaussian_split_suggestions);
    HT_RUN_TEST(test_gaussian_binary_split_dists);

    /* Nível 1: Histogram Non-Parametric Observer */
    HT_RUN_TEST(test_hist_obs_basic_and_merging);
    HT_RUN_TEST(test_hist_obs_split_dists_and_proba);

    /* Nível 1: Categorical Nominal & Candidates */
    HT_RUN_TEST(test_categorical_basic_and_proba);
    HT_RUN_TEST(test_categorical_edge_cases);
    HT_RUN_TEST(test_categorical_binary_split_dists);
    HT_RUN_TEST(test_split_candidate_struct);

    /* Nível 1: Variância Incremental & Observadores de Regressão */
    HT_RUN_TEST(test_var_stat_init_and_welford_analytical);
    HT_RUN_TEST(test_var_stat_chan_combine_analytical);
    HT_RUN_TEST(test_var_stat_chan_subtract_analytical);
    HT_RUN_TEST(test_reg_numeric_obs_ordered_slots_and_merging);
    HT_RUN_TEST(test_reg_categorical_obs_basic_and_edge_cases);

    /* Nível 1: Fusão de Estatísticas Suficientes (Federado / Paralelo) */
    HT_RUN_TEST(test_observer_gaussian_and_categorical_merge_exact);
    HT_RUN_TEST(test_observer_hist_and_reg_merge);

    /* Nível 2: Integração com ht_memory slab pool & Block-Rescaling */
    HT_RUN_TEST(test_observers_in_leaf_stats_pool);
    HT_RUN_TEST(test_reg_observers_in_leaf_stats_pool);
    HT_RUN_TEST(test_observer_100k_stream_block_rescaling);
    HT_RUN_TEST(test_var_stat_high_amplitude_regression_q24_8);
}
