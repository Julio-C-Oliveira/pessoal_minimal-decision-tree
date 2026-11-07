/**
 * @file test_observer.c
 * @brief Level 1 & 2 Unit and Integration tests for Attribute Observers (ht_observer).
 */

#include "test_harness.h"
#include "ht_observer.h"
#include "ht_memory.h"
#include <math.h>

/* ========================================================================= */
/* --- Level 1: Gaussian Observer Unit Tests ------------------------------- */
/* ========================================================================= */

static void test_gaussian_init(void) {
    ht_gaussian_obs_t obs;
    obs.count = (ht_float_t)10.0;
    obs.mean = (ht_float_t)5.0;
    obs.m2 = (ht_float_t)2.0;
    obs.min_val = (ht_float_t)1.0;
    obs.max_val = (ht_float_t)9.0;

    ht_gaussian_init(&obs);

    HT_ASSERT_FLOAT_EQ(obs.count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(obs.mean, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(obs.m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(obs.min_val, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(obs.max_val, (ht_float_t)0.0);
}

static void test_gaussian_welford_analytical(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    /* Known dataset: {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}
     * n = 8
     * sum = 40.0
     * mean = 5.0
     * squared diffs: (-3)^2 + (-1)^2 * 3 + 0^2 * 2 + 2^2 + 4^2 = 9 + 3 + 0 + 4 + 16 = 32.0
     * M2 = 32.0
     * sample variance s^2 = 32.0 / 7.0 = 4.57142857...
     * sample stddev = sqrt(32/7) = 2.1380899...
     * min = 2.0, max = 9.0
     */
    const ht_float_t samples[] = {
        (ht_float_t)2.0, (ht_float_t)4.0, (ht_float_t)4.0, (ht_float_t)4.0,
        (ht_float_t)5.0, (ht_float_t)5.0, (ht_float_t)7.0, (ht_float_t)9.0
    };
    const uint8_t n_samples = sizeof(samples) / sizeof(samples[0]);

    for (uint8_t i = 0; i < n_samples; ++i) {
        ht_gaussian_update(obs, 0, samples[i], (ht_float_t)1.0);
    }

    HT_ASSERT_FLOAT_EQ(obs[0].count, (ht_float_t)8.0);
    HT_ASSERT_FLOAT_NEAR(obs[0].mean, (ht_float_t)5.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs[0].m2, (ht_float_t)32.0, 1e-4);
    HT_ASSERT_FLOAT_EQ(obs[0].min_val, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_EQ(obs[0].max_val, (ht_float_t)9.0);

    ht_float_t expected_var = (ht_float_t)(32.0 / 7.0);
    ht_float_t calc_var = ht_gaussian_variance(&obs[0]);
    HT_ASSERT_FLOAT_NEAR(calc_var, expected_var, 1e-4);

    ht_float_t expected_std = HT_SQRT(expected_var);
    ht_float_t calc_std = ht_gaussian_stddev(&obs[0]);
    HT_ASSERT_FLOAT_NEAR(calc_std, expected_std, 1e-4);
}

static void test_gaussian_welford_weighted(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    /* Two weighted samples:
     * x1 = 10.0, w1 = 2.0
     * x2 = 20.0, w2 = 2.0
     * Total weight W = 4.0
     * Mean = 15.0
     * M2 = 2*(10 - 15)^2 + 2*(20 - 15)^2 = 2*25 + 2*25 = 100.0
     * Unbiased variance s^2 = 100.0 / (4.0 - 1.0) = 33.333333...
     */
    ht_gaussian_update(obs, 1, (ht_float_t)10.0, (ht_float_t)2.0);
    ht_gaussian_update(obs, 1, (ht_float_t)20.0, (ht_float_t)2.0);

    HT_ASSERT_FLOAT_EQ(obs[1].count, (ht_float_t)4.0);
    HT_ASSERT_FLOAT_NEAR(obs[1].mean, (ht_float_t)15.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs[1].m2, (ht_float_t)100.0, 1e-4);

    ht_float_t expected_var = (ht_float_t)(100.0 / 3.0);
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&obs[1]), expected_var, 1e-4);
}

static void test_gaussian_edge_cases(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    /* Edge Case 1: n = 0 */
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs[0], (ht_float_t)5.0), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs[0], (ht_float_t)5.0), -HT_INFINITY);

    /* Edge Case 2: n = 1 */
    ht_gaussian_update(obs, 0, (ht_float_t)42.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs[0].count, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs[0].mean, (ht_float_t)42.0);
    HT_ASSERT_FLOAT_EQ(obs[0].m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs[0], (ht_float_t)42.0), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs[0], (ht_float_t)42.0), -HT_INFINITY);

    /* Edge Case 3: Constant feature (variance = 0, n = 3) */
    ht_gaussian_update(obs, 0, (ht_float_t)42.0, (ht_float_t)1.0);
    ht_gaussian_update(obs, 0, (ht_float_t)42.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs[0].count, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_EQ(obs[0].m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs[0], (ht_float_t)42.0), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs[0], (ht_float_t)42.0), -HT_INFINITY);

    /* Edge Case 4: NaN input value must be ignored */
    ht_float_t nan_val = (ht_float_t)0.0 / (ht_float_t)0.0;
    ht_gaussian_update(obs, 0, nan_val, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs[0].count, (ht_float_t)3.0); /* Count unchanged */

    /* Edge Case 5: Non-positive weight must be ignored */
    ht_gaussian_update(obs, 0, (ht_float_t)100.0, (ht_float_t)0.0);
    ht_gaussian_update(obs, 0, (ht_float_t)100.0, (ht_float_t)-2.0);
    HT_ASSERT_FLOAT_EQ(obs[0].count, (ht_float_t)3.0);

    /* Edge Case 6: Out of bounds class_idx */
    ht_gaussian_update(obs, HT_MAX_CLASSES, (ht_float_t)100.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs[0].count, (ht_float_t)3.0);

    /* Edge Case 7: NULL observer pointer */
    ht_gaussian_update(NULL, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(NULL), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(NULL), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(NULL, (ht_float_t)1.0), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(NULL, (ht_float_t)1.0), -HT_INFINITY);
}

static void test_normal_cdf(void) {
    /* Mean = 0, Stddev = 1: Standard normal Phi(x) */
    ht_float_t mean = (ht_float_t)0.0;
    ht_float_t std = (ht_float_t)1.0;

    /* Phi(0) = 0.5 */
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf((ht_float_t)0.0, mean, std), (ht_float_t)0.5, 1e-6);

    /* Phi(1) approx 0.8413447 */
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf((ht_float_t)1.0, mean, std), (ht_float_t)0.8413447, 1e-5);

    /* Phi(-1) approx 0.1586553 */
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf((ht_float_t)-1.0, mean, std), (ht_float_t)0.1586553, 1e-5);

    /* Symmetry: Phi(-x) + Phi(x) == 1.0 */
    ht_float_t cdf_pos = ht_normal_cdf((ht_float_t)1.5, mean, std);
    ht_float_t cdf_neg = ht_normal_cdf((ht_float_t)-1.5, mean, std);
    HT_ASSERT_FLOAT_NEAR(cdf_pos + cdf_neg, (ht_float_t)1.0, 1e-5);

    /* 95% interval: Phi(1.95996) approx 0.975 */
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf((ht_float_t)1.95996398454, mean, std), (ht_float_t)0.975, 1e-4);

    /* Extreme tails: clamped within [0, 1] */
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf((ht_float_t)10.0, mean, std), (ht_float_t)1.0, 1e-6);
    HT_ASSERT_FLOAT_NEAR(ht_normal_cdf((ht_float_t)-10.0, mean, std), (ht_float_t)0.0, 1e-6);

    /* Degenerate cases: std <= 0 */
    HT_ASSERT_FLOAT_EQ(ht_normal_cdf((ht_float_t)4.0, (ht_float_t)5.0, (ht_float_t)0.0), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_normal_cdf((ht_float_t)6.0, (ht_float_t)5.0, (ht_float_t)0.0), (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(ht_normal_cdf((ht_float_t)5.0, (ht_float_t)5.0, (ht_float_t)0.0), (ht_float_t)0.5);

    /* NaN propagation */
    ht_float_t nan_val = (ht_float_t)0.0 / (ht_float_t)0.0;
    HT_ASSERT_NAN(ht_normal_cdf(nan_val, mean, std));
    HT_ASSERT_NAN(ht_normal_cdf((ht_float_t)0.0, nan_val, std));
}

static void test_gaussian_pdf_and_log_pdf(void) {
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

    /* Setup a standard normal distribution: mu = 0, s^2 = 1 (m2 = 99, n = 100) */
    obs.count = (ht_float_t)100.0;
    obs.mean = (ht_float_t)0.0;
    obs.m2 = (ht_float_t)99.0; /* variance = 99 / 99 = 1.0 */
    obs.min_val = (ht_float_t)-3.0;
    obs.max_val = (ht_float_t)3.0;

    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&obs), (ht_float_t)1.0, 1e-6);

    /* Normal PDF at x = 0: p(0) = 1 / sqrt(2*pi) approx 0.3989422804 */
    ht_float_t expected_pdf_0 = (ht_float_t)0.3989422804;
    ht_float_t calc_pdf_0 = ht_gaussian_cond_proba(&obs, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_NEAR(calc_pdf_0, expected_pdf_0, 1e-5);

    /* Log-PDF at x = 0: ln(1 / sqrt(2*pi)) = -0.5 * ln(2*pi) approx -0.918938533 */
    ht_float_t expected_log_pdf_0 = (ht_float_t)-0.918938533;
    ht_float_t calc_log_pdf_0 = ht_gaussian_cond_log_proba(&obs, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_NEAR(calc_log_pdf_0, expected_log_pdf_0, 1e-5);

    /* Consistency: exp(log_pdf) == pdf */
    HT_ASSERT_FLOAT_NEAR(HT_EXP(calc_log_pdf_0), calc_pdf_0, 1e-5);

    /* Normal PDF at x = 1: p(1) = (1 / sqrt(2*pi)) * exp(-0.5) approx 0.2419707245 */
    ht_float_t expected_pdf_1 = (ht_float_t)0.2419707245;
    ht_float_t calc_pdf_1 = ht_gaussian_cond_proba(&obs, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_NEAR(calc_pdf_1, expected_pdf_1, 1e-5);

    /* Log-PDF at x = 1: -0.918938533 - 0.5 = -1.418938533 */
    ht_float_t expected_log_pdf_1 = (ht_float_t)-1.418938533;
    ht_float_t calc_log_pdf_1 = ht_gaussian_cond_log_proba(&obs, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_NEAR(calc_log_pdf_1, expected_log_pdf_1, 1e-5);
    HT_ASSERT_FLOAT_NEAR(HT_EXP(calc_log_pdf_1), calc_pdf_1, 1e-5);
}

static void test_gaussian_split_suggestions(void) {
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&obs[c]);
    }

    /* Class 0 observed in range [10.0, 20.0] */
    ht_gaussian_update(obs, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_gaussian_update(obs, 0, (ht_float_t)20.0, (ht_float_t)1.0);

    /* Class 1 observed in range [15.0, 32.0] */
    ht_gaussian_update(obs, 1, (ht_float_t)15.0, (ht_float_t)1.0);
    ht_gaussian_update(obs, 1, (ht_float_t)32.0, (ht_float_t)1.0);

    /* Overall range: [10.0, 32.0], delta = 22.0
     * With HT_GAUSSIAN_SPLITS = 10, bin_size = 22.0 / 11.0 = 2.0
     * Suggested splits: 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0
     */
    ht_float_t splits[HT_GAUSSIAN_SPLITS];
    uint8_t n_splits = ht_gaussian_split_suggestions(obs, 2, splits);

    HT_ASSERT_EQ(n_splits, HT_GAUSSIAN_SPLITS);

    ht_float_t prev = (ht_float_t)10.0;
    for (uint8_t i = 0; i < n_splits; ++i) {
        /* Strictly increasing and strictly within (10.0, 32.0) */
        HT_ASSERT(splits[i] > prev);
        HT_ASSERT(splits[i] < (ht_float_t)32.0);
        ht_float_t expected = (ht_float_t)(10.0 + 2.0 * (i + 1));
        HT_ASSERT_FLOAT_NEAR(splits[i], expected, 1e-4);
        prev = splits[i];
    }

    /* Constant feature across all classes: 0 splits */
    ht_gaussian_obs_t const_obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        ht_gaussian_init(&const_obs[c]);
    }
    ht_gaussian_update(const_obs, 0, (ht_float_t)5.0, (ht_float_t)10.0);
    ht_gaussian_update(const_obs, 1, (ht_float_t)5.0, (ht_float_t)10.0);

    uint8_t n_const_splits = ht_gaussian_split_suggestions(const_obs, 2, splits);
    HT_ASSERT_EQ(n_const_splits, 0);

    /* Empty observer: 0 splits */
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

    /* Setup class 0: range [10, 20], mean = 15, count = 100 */
    obs[0].count = (ht_float_t)100.0;
    obs[0].mean = (ht_float_t)15.0;
    obs[0].m2 = (ht_float_t)(99.0 * 4.0); /* variance = 4.0, std = 2.0 */
    obs[0].min_val = (ht_float_t)10.0;
    obs[0].max_val = (ht_float_t)20.0;

    /* Setup class 1: range [25, 35], mean = 30, count = 50 */
    obs[1].count = (ht_float_t)50.0;
    obs[1].mean = (ht_float_t)30.0;
    obs[1].m2 = (ht_float_t)(49.0 * 1.0); /* variance = 1.0, std = 1.0 */
    obs[1].min_val = (ht_float_t)25.0;
    obs[1].max_val = (ht_float_t)35.0;

    ht_float_t lhs[HT_MAX_CLASSES];
    ht_float_t rhs[HT_MAX_CLASSES];

    /* Split 1: t = 5.0 (below all min_vals) */
    ht_gaussian_binary_split_dists(obs, 2, (ht_float_t)5.0, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(rhs[0], (ht_float_t)100.0);
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)50.0);

    /* Split 2: t = 40.0 (above all max_vals) */
    ht_gaussian_binary_split_dists(obs, 2, (ht_float_t)40.0, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], (ht_float_t)100.0);
    HT_ASSERT_FLOAT_EQ(rhs[0], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)50.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)0.0);

    /* Split 3: t = 15.0 (at mean of class 0, below min of class 1) */
    ht_gaussian_binary_split_dists(obs, 2, (ht_float_t)15.0, lhs, rhs);
    /* For class 0: Phi(15) = 0.5 -> lhs = 50, rhs = 50 */
    HT_ASSERT_FLOAT_NEAR(lhs[0], (ht_float_t)50.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(rhs[0], (ht_float_t)50.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(lhs[0] + rhs[0], (ht_float_t)100.0, 1e-4);
    /* For class 1: 15 < min_val (25) -> lhs = 0, rhs = 50 */
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)50.0);

    /* Split 4: t = 22.0 (above max of class 0, below min of class 1) */
    ht_gaussian_binary_split_dists(obs, 2, (ht_float_t)22.0, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], (ht_float_t)100.0);
    HT_ASSERT_FLOAT_EQ(rhs[0], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)50.0);
}

/* ========================================================================= */
/* --- Level 1: Categorical Nominal Observer Unit Tests ------------------- */
/* ========================================================================= */

static void test_categorical_basic_and_proba(void) {
    ht_categorical_obs_t cat;
    ht_categorical_init(&cat);

    /* Class 0: cat 0 with w=3.0, cat 1 with w=1.0. Total = 4.0 */
    ht_categorical_update(&cat, 0, 0, (ht_float_t)3.0);
    ht_categorical_update(&cat, 0, 1, (ht_float_t)1.0);

    /* Class 1: cat 1 with w=2.0, cat 2 with w=8.0. Total = 10.0 */
    ht_categorical_update(&cat, 1, 1, (ht_float_t)2.0);
    ht_categorical_update(&cat, 1, 2, (ht_float_t)8.0);

    /* Class 0 probabilities */
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 0, 0), (ht_float_t)0.75, 1e-6);
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 0, 1), (ht_float_t)0.25, 1e-6);
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 0, 2), (ht_float_t)0.0);

    /* Class 1 probabilities */
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 1, 0), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 1, 1), (ht_float_t)0.20, 1e-6);
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&cat, 1, 2), (ht_float_t)0.80, 1e-6);

    /* Category observation check */
    HT_ASSERT(ht_categorical_is_observed(&cat, 0));
    HT_ASSERT(ht_categorical_is_observed(&cat, 1));
    HT_ASSERT(ht_categorical_is_observed(&cat, 2));
    HT_ASSERT(!ht_categorical_is_observed(&cat, 3));
    HT_ASSERT(!ht_categorical_is_observed(&cat, 4));
}

static void test_categorical_edge_cases(void) {
    ht_categorical_obs_t cat;
    ht_categorical_init(&cat);

    /* Negative category */
    ht_categorical_update(&cat, 0, -1, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 0, -1), (ht_float_t)0.0);

    /* Out of bounds category (>= HT_MAX_CATEGORIES) */
    ht_categorical_update(&cat, 0, HT_MAX_CATEGORIES, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 0, HT_MAX_CATEGORIES), (ht_float_t)0.0);

    /* Out of bounds class_idx */
    ht_categorical_update(&cat, HT_MAX_CLASSES, 0, (ht_float_t)1.0);

    /* Non-positive weight */
    ht_categorical_update(&cat, 0, 0, (ht_float_t)0.0);
    ht_categorical_update(&cat, 0, 0, (ht_float_t)-5.0);

    /* Verify still completely empty */
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            HT_ASSERT_FLOAT_EQ(cat.counts[c][v], (ht_float_t)0.0);
        }
    }

    /* Querying unobserved class returns 0.0 */
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(&cat, 2, 0), (ht_float_t)0.0);
    HT_ASSERT(!ht_categorical_is_observed(&cat, 0));

    /* NULL pointers */
    ht_categorical_init(NULL);
    ht_categorical_update(NULL, 0, 0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(ht_categorical_cond_proba(NULL, 0, 0), (ht_float_t)0.0);
    HT_ASSERT(!ht_categorical_is_observed(NULL, 0));
}

static void test_categorical_binary_split_dists(void) {
    ht_categorical_obs_t cat;
    ht_categorical_init(&cat);

    /* Class 0: cat 0 (weight 3.0), cat 1 (weight 1.0) */
    ht_categorical_update(&cat, 0, 0, (ht_float_t)3.0);
    ht_categorical_update(&cat, 0, 1, (ht_float_t)1.0);

    /* Class 1: cat 1 (weight 2.0), cat 2 (weight 8.0) */
    ht_categorical_update(&cat, 1, 1, (ht_float_t)2.0);
    ht_categorical_update(&cat, 1, 2, (ht_float_t)8.0);

    ht_float_t lhs[HT_MAX_CLASSES];
    ht_float_t rhs[HT_MAX_CLASSES];

    /* Binary split on cat == 0 (left: x == 0, right: x != 0) */
    ht_categorical_binary_split_dists(&cat, 2, 0, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], (ht_float_t)3.0);
    HT_ASSERT_FLOAT_EQ(rhs[0], (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)10.0);

    /* Binary split on cat == 1 (left: x == 1, right: x != 1) */
    ht_categorical_binary_split_dists(&cat, 2, 1, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(rhs[0], (ht_float_t)3.0);
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)2.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)8.0);

    /* Conservation of total weight */
    for (uint8_t c = 0; c < 2; ++c) {
        HT_ASSERT_FLOAT_EQ(lhs[c] + rhs[c], (c == 0) ? (ht_float_t)4.0 : (ht_float_t)10.0);
    }
}

static void test_split_candidate_struct(void) {
#ifndef HT_USE_DOUBLE
    HT_ASSERT_EQ(sizeof(ht_split_candidate_t), 44);
#else
    HT_ASSERT_EQ(sizeof(ht_split_candidate_t), 88);
#endif

    ht_split_candidate_t cand;
    cand.merit = (ht_float_t)123.45;
    cand.feature_idx = 5;
    cand.is_categorical = 1;
    cand.split_val.threshold = (ht_float_t)42.0;

    ht_split_candidate_init(&cand);

    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);
    HT_ASSERT_EQ(cand.feature_idx, HT_NO_FEATURE);
    HT_ASSERT_EQ(cand.is_categorical, 0);
    HT_ASSERT_FLOAT_EQ(cand.split_val.threshold, (ht_float_t)0.0);

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        HT_ASSERT_FLOAT_EQ(cand.lhs_dist[c], (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(cand.rhs_dist[c], (ht_float_t)0.0);
    }
}

/* ========================================================================= */
/* --- Level 2: Meso Integration Tests (Observers in Pool Memory) ---------- */
/* ========================================================================= */

static ht_memory_pool_t s_observer_test_pool;

static void test_observers_in_leaf_stats_pool(void) {
    ht_memory_init(&s_observer_test_pool);

    /* Allocate a leaf statistics block from the deterministic slab allocator */
    uint16_t stats_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);

    ht_leaf_stats_t *stats = ht_stats_get(&s_observer_test_pool, stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    /* All observers inside freshly allocated leaf_stats must be zero-initialized */
    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].count, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].mean, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].m2, (ht_float_t)0.0);
        }
    }

    /* Feature 0 is numeric: update Gaussian observers */
    ht_gaussian_update(stats->observers[0].num, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_gaussian_update(stats->observers[0].num, 0, (ht_float_t)20.0, (ht_float_t)1.0);
    ht_gaussian_update(stats->observers[0].num, 0, (ht_float_t)30.0, (ht_float_t)1.0);

    HT_ASSERT_FLOAT_EQ(stats->observers[0].num[0].count, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_NEAR(stats->observers[0].num[0].mean, (ht_float_t)20.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&stats->observers[0].num[0]), (ht_float_t)100.0, 1e-4);

    /* Feature 1 is categorical: update Nominal observers */
    ht_categorical_update(&stats->observers[1].cat, 0, 2, (ht_float_t)5.0);
    ht_categorical_update(&stats->observers[1].cat, 0, 4, (ht_float_t)15.0);

    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&stats->observers[1].cat, 0, 2), (ht_float_t)0.25, 1e-5);
    HT_ASSERT_FLOAT_NEAR(ht_categorical_cond_proba(&stats->observers[1].cat, 0, 4), (ht_float_t)0.75, 1e-5);

    /* Free leaf stats block back to the pool */
    ht_stats_free(&s_observer_test_pool, stats_idx);
    HT_ASSERT_EQ(ht_stats_count(&s_observer_test_pool), 0);

    /* Reallocate: free-list must return same stats_idx, freshly zeroed */
    uint16_t recycled_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_EQ(recycled_idx, stats_idx);

    ht_leaf_stats_t *recycled_stats = ht_stats_get(&s_observer_test_pool, recycled_idx);
    HT_ASSERT_NOT_NULL(recycled_stats);
    HT_ASSERT_FLOAT_EQ(recycled_stats->observers[0].num[0].count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(recycled_stats->observers[1].cat.counts[0][2], (ht_float_t)0.0);
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

    /* Nível 1: Categorical Nominal & Candidates */
    HT_RUN_TEST(test_categorical_basic_and_proba);
    HT_RUN_TEST(test_categorical_edge_cases);
    HT_RUN_TEST(test_categorical_binary_split_dists);
    HT_RUN_TEST(test_split_candidate_struct);

    /* Nível 2: Integração com ht_memory slab pool */
    HT_RUN_TEST(test_observers_in_leaf_stats_pool);
}
