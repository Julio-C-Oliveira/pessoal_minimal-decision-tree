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
    obs.classes[0].count = (ht_float_t)10.0;
    obs.classes[0].mean = (ht_float_t)5.0;
    obs.classes[0].m2 = (ht_float_t)2.0;
    obs.classes[0].min_val = (ht_float_t)1.0;
    obs.classes[0].max_val = (ht_float_t)9.0;

    ht_gaussian_init(&obs);

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        HT_ASSERT_FLOAT_EQ(obs.classes[c].count, (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(obs.classes[c].mean, (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(obs.classes[c].m2, (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(obs.classes[c].min_val, (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(obs.classes[c].max_val, (ht_float_t)0.0);
    }
}

static void test_gaussian_welford_analytical(void) {
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

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
        ht_gaussian_update(&obs, 0, samples[i], (ht_float_t)1.0);
    }

    HT_ASSERT_FLOAT_EQ(obs.classes[0].count, (ht_float_t)8.0);
    HT_ASSERT_FLOAT_NEAR(obs.classes[0].mean, (ht_float_t)5.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.classes[0].m2, (ht_float_t)32.0, 1e-4);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].min_val, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].max_val, (ht_float_t)9.0);

    ht_float_t expected_var = (ht_float_t)(32.0 / 7.0);
    ht_float_t calc_var = ht_gaussian_variance(&obs.classes[0]);
    HT_ASSERT_FLOAT_NEAR(calc_var, expected_var, 1e-4);

    ht_float_t expected_std = HT_SQRT(expected_var);
    ht_float_t calc_std = ht_gaussian_stddev(&obs.classes[0]);
    HT_ASSERT_FLOAT_NEAR(calc_std, expected_std, 1e-4);
}

static void test_gaussian_welford_weighted(void) {
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

    /* Two weighted samples:
     * x1 = 10.0, w1 = 2.0
     * x2 = 20.0, w2 = 2.0
     * Total weight W = 4.0
     * Mean = 15.0
     * M2 = 2*(10 - 15)^2 + 2*(20 - 15)^2 = 2*25 + 2*25 = 100.0
     * Unbiased variance s^2 = 100.0 / (4.0 - 1.0) = 33.333333...
     */
    ht_gaussian_update(&obs, 1, (ht_float_t)10.0, (ht_float_t)2.0);
    ht_gaussian_update(&obs, 1, (ht_float_t)20.0, (ht_float_t)2.0);

    HT_ASSERT_FLOAT_EQ(obs.classes[1].count, (ht_float_t)4.0);
    HT_ASSERT_FLOAT_NEAR(obs.classes[1].mean, (ht_float_t)15.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.classes[1].m2, (ht_float_t)100.0, 1e-4);

    ht_float_t expected_var = (ht_float_t)(100.0 / 3.0);
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&obs.classes[1]), expected_var, 1e-4);
}

static void test_gaussian_edge_cases(void) {
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

    /* Edge Case 1: n = 0 */
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs.classes[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs.classes[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs.classes[0], (ht_float_t)5.0), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs.classes[0], (ht_float_t)5.0), -HT_INFINITY);

    /* Edge Case 2: n = 1 */
    ht_gaussian_update(&obs, 0, (ht_float_t)42.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].count, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].mean, (ht_float_t)42.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs.classes[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs.classes[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs.classes[0], (ht_float_t)42.0), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs.classes[0], (ht_float_t)42.0), -HT_INFINITY);

    /* Edge Case 3: Constant feature (variance = 0, n = 3) */
    ht_gaussian_update(&obs, 0, (ht_float_t)42.0, (ht_float_t)1.0);
    ht_gaussian_update(&obs, 0, (ht_float_t)42.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].count, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_variance(&obs.classes[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_stddev(&obs.classes[0]), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_gaussian_cond_proba(&obs.classes[0], (ht_float_t)42.0), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_gaussian_cond_log_proba(&obs.classes[0], (ht_float_t)42.0), -HT_INFINITY);

    /* Edge Case 4: NaN input value must be ignored */
    ht_float_t nan_val = (ht_float_t)0.0 / (ht_float_t)0.0;
    ht_gaussian_update(&obs, 0, nan_val, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].count, (ht_float_t)3.0); /* Count unchanged */

    /* Edge Case 5: Non-positive weight must be ignored */
    ht_gaussian_update(&obs, 0, (ht_float_t)100.0, (ht_float_t)0.0);
    ht_gaussian_update(&obs, 0, (ht_float_t)100.0, (ht_float_t)-2.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].count, (ht_float_t)3.0);

    /* Edge Case 6: Out of bounds class_idx */
    ht_gaussian_update(&obs, HT_MAX_CLASSES, (ht_float_t)100.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(obs.classes[0].count, (ht_float_t)3.0);

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
    ht_gaussian_class_stat_t stat;
    stat.count = (ht_float_t)100.0;
    stat.mean = (ht_float_t)0.0;
    stat.m2 = (ht_float_t)99.0; /* variance = 99 / 99 = 1.0 */

    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&stat), (ht_float_t)1.0, 1e-6);

    /* Normal PDF at x = 0: p(0) = 1 / sqrt(2*pi) approx 0.3989422804 */
    ht_float_t expected_pdf_0 = (ht_float_t)0.3989422804;
    ht_float_t calc_pdf_0 = ht_gaussian_cond_proba(&stat, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_NEAR(calc_pdf_0, expected_pdf_0, 1e-5);

    /* Log-PDF at x = 0: ln(1 / sqrt(2*pi)) = -0.5 * ln(2*pi) approx -0.918938533 */
    ht_float_t expected_log_pdf_0 = (ht_float_t)-0.918938533;
    ht_float_t calc_log_pdf_0 = ht_gaussian_cond_log_proba(&stat, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_NEAR(calc_log_pdf_0, expected_log_pdf_0, 1e-5);

    /* Consistency: exp(log_pdf) == pdf */
    HT_ASSERT_FLOAT_NEAR(HT_EXP(calc_log_pdf_0), calc_pdf_0, 1e-5);

    /* Normal PDF at x = 1: p(1) = (1 / sqrt(2*pi)) * exp(-0.5) approx 0.2419707245 */
    ht_float_t expected_pdf_1 = (ht_float_t)0.2419707245;
    ht_float_t calc_pdf_1 = ht_gaussian_cond_proba(&stat, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_NEAR(calc_pdf_1, expected_pdf_1, 1e-5);

    /* Log-PDF at x = 1: -0.918938533 - 0.5 = -1.418938533 */
    ht_float_t expected_log_pdf_1 = (ht_float_t)-1.418938533;
    ht_float_t calc_log_pdf_1 = ht_gaussian_cond_log_proba(&stat, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_NEAR(calc_log_pdf_1, expected_log_pdf_1, 1e-5);
    HT_ASSERT_FLOAT_NEAR(HT_EXP(calc_log_pdf_1), calc_pdf_1, 1e-5);
}

static void test_gaussian_split_suggestions(void) {
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

    /* Class 0 observed in range [10.0, 20.0] */
    ht_gaussian_update(&obs, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_gaussian_update(&obs, 0, (ht_float_t)20.0, (ht_float_t)1.0);

    /* Class 1 observed in range [15.0, 32.0] */
    ht_gaussian_update(&obs, 1, (ht_float_t)15.0, (ht_float_t)1.0);
    ht_gaussian_update(&obs, 1, (ht_float_t)32.0, (ht_float_t)1.0);

    /* Overall range: [10.0, 32.0], delta = 22.0
     * With HT_GAUSSIAN_SPLITS = 10, bin_size = 22.0 / 11.0 = 2.0
     * Suggested splits: 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0
     */
    ht_float_t splits[HT_GAUSSIAN_SPLITS];
    uint8_t n_splits = ht_gaussian_split_suggestions(&obs, 2, splits);

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
    ht_gaussian_obs_t const_obs;
    ht_gaussian_init(&const_obs);
    ht_gaussian_update(&const_obs, 0, (ht_float_t)5.0, (ht_float_t)10.0);
    ht_gaussian_update(&const_obs, 1, (ht_float_t)5.0, (ht_float_t)10.0);

    uint8_t n_const_splits = ht_gaussian_split_suggestions(&const_obs, 2, splits);
    HT_ASSERT_EQ(n_const_splits, 0);

    /* Empty observer: 0 splits */
    ht_gaussian_obs_t empty_obs;
    ht_gaussian_init(&empty_obs);
    HT_ASSERT_EQ(ht_gaussian_split_suggestions(&empty_obs, 2, splits), 0);
}

static void test_gaussian_binary_split_dists(void) {
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

    /* Setup class 0: range [10, 20], mean = 15, count = 100 */
    obs.classes[0].count = (ht_float_t)100.0;
    obs.classes[0].mean = (ht_float_t)15.0;
    obs.classes[0].m2 = (ht_float_t)(99.0 * 4.0); /* variance = 4.0, std = 2.0 */
    obs.classes[0].min_val = (ht_float_t)10.0;
    obs.classes[0].max_val = (ht_float_t)20.0;

    /* Setup class 1: range [25, 35], mean = 30, count = 50 */
    obs.classes[1].count = (ht_float_t)50.0;
    obs.classes[1].mean = (ht_float_t)30.0;
    obs.classes[1].m2 = (ht_float_t)(49.0 * 1.0); /* variance = 1.0, std = 1.0 */
    obs.classes[1].min_val = (ht_float_t)25.0;
    obs.classes[1].max_val = (ht_float_t)35.0;

    ht_float_t lhs[HT_MAX_CLASSES];
    ht_float_t rhs[HT_MAX_CLASSES];

    /* Split 1: t = 5.0 (below min_val) */
    ht_gaussian_binary_split_dists(&obs, 2, (ht_float_t)5.0, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(rhs[0], (ht_float_t)100.0);
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)50.0);

    /* Split 2: t = 40.0 (above max_val) */
    ht_gaussian_binary_split_dists(&obs, 2, (ht_float_t)40.0, lhs, rhs);
    HT_ASSERT_FLOAT_EQ(lhs[0], (ht_float_t)100.0);
    HT_ASSERT_FLOAT_EQ(rhs[0], (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(lhs[1], (ht_float_t)50.0);
    HT_ASSERT_FLOAT_EQ(rhs[1], (ht_float_t)0.0);

    /* Split 3: t = 15.0 (at mean of class 0, far below class 1) */
    ht_gaussian_binary_split_dists(&obs, 2, (ht_float_t)15.0, lhs, rhs);
    /* For class 0: Phi(15) = 0.5 -> lhs = 50, rhs = 50 */
    HT_ASSERT_FLOAT_NEAR(lhs[0], (ht_float_t)50.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(rhs[0], (ht_float_t)50.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(lhs[0] + rhs[0], (ht_float_t)100.0, 1e-4);
    /* For class 1: 15 is 15 std away -> lhs = 0, rhs = 50 */
    HT_ASSERT_FLOAT_NEAR(lhs[1], (ht_float_t)0.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(rhs[1], (ht_float_t)50.0, 1e-4);

    /* Split 4: t = 22.0 (between class 0 and class 1) */
    ht_gaussian_binary_split_dists(&obs, 2, (ht_float_t)22.0, lhs, rhs);
    HT_ASSERT_FLOAT_NEAR(lhs[0], (ht_float_t)100.0, 0.1);
    HT_ASSERT_FLOAT_NEAR(rhs[0], (ht_float_t)0.0, 0.1);
    HT_ASSERT_FLOAT_NEAR(lhs[1], (ht_float_t)0.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(rhs[1], (ht_float_t)50.0, 1e-4);
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
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num.classes[c].count, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num.classes[c].mean, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num.classes[c].m2, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num.classes[c].min_val, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num.classes[c].max_val, (ht_float_t)0.0);
        }
    }

    /* Feature 0 is numeric: update Gaussian observers */
    ht_gaussian_update(&stats->observers[0].num, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_gaussian_update(&stats->observers[0].num, 0, (ht_float_t)20.0, (ht_float_t)1.0);
    ht_gaussian_update(&stats->observers[0].num, 0, (ht_float_t)30.0, (ht_float_t)1.0);

    HT_ASSERT_FLOAT_EQ(stats->observers[0].num.classes[0].count, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_NEAR(stats->observers[0].num.classes[0].mean, (ht_float_t)20.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(ht_gaussian_variance(&stats->observers[0].num.classes[0]), (ht_float_t)100.0, 1e-4);

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
    HT_ASSERT_FLOAT_EQ(recycled_stats->observers[0].num.classes[0].count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(recycled_stats->observers[1].cat.counts[0][2], (ht_float_t)0.0);
}

/* ========================================================================= */
/* --- Level 1 & 2: Incremental Variance & Regression Observers ------------ */
/* ========================================================================= */

static void test_var_stat_init_and_welford_analytical(void) {
    ht_var_stat_t v;
    v.count = (ht_float_t)10.0;
    v.mean = (ht_float_t)5.0;
    v.m2 = (ht_float_t)2.0;

    ht_var_init(&v);
    HT_ASSERT_FLOAT_EQ(v.count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(v.mean, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(v.m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_var_value(&v), (ht_float_t)0.0);

    /* Edge cases: NULL pointer */
    ht_var_init(NULL);
    ht_var_update(NULL, (ht_float_t)1.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(ht_var_value(NULL), (ht_float_t)0.0);

    /* Edge cases: n = 1 */
    ht_var_update(&v, (ht_float_t)42.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(v.count, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(v.mean, (ht_float_t)42.0);
    HT_ASSERT_FLOAT_EQ(v.m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_var_value(&v), (ht_float_t)0.0);

    /* Edge cases: constant target (n = 3, variance = 0) */
    ht_var_update(&v, (ht_float_t)42.0, (ht_float_t)1.0);
    ht_var_update(&v, (ht_float_t)42.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(v.count, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_EQ(v.m2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_var_value(&v), (ht_float_t)0.0);

    /* Edge cases: NaN, infinite, non-positive weight */
    ht_float_t nan_val = (ht_float_t)0.0 / (ht_float_t)0.0;
    ht_var_update(&v, nan_val, (ht_float_t)1.0);
    ht_var_update(&v, (ht_float_t)10.0, (ht_float_t)0.0);
    ht_var_update(&v, (ht_float_t)10.0, (ht_float_t)-1.0);
    HT_ASSERT_FLOAT_EQ(v.count, (ht_float_t)3.0);

    /* Analytical Welford on known dataset: {2, 4, 4, 4, 5, 5, 7, 9}
     * n = 8, mean = 5.0, M2 = 32.0, s^2 = 32 / 7 = 4.57142857
     */
    ht_var_init(&v);
    const ht_float_t samples[] = {
        (ht_float_t)2.0, (ht_float_t)4.0, (ht_float_t)4.0, (ht_float_t)4.0,
        (ht_float_t)5.0, (ht_float_t)5.0, (ht_float_t)7.0, (ht_float_t)9.0
    };
    for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        ht_var_update(&v, samples[i], (ht_float_t)1.0);
    }
    HT_ASSERT_FLOAT_EQ(v.count, (ht_float_t)8.0);
    HT_ASSERT_FLOAT_NEAR(v.mean, (ht_float_t)5.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(v.m2, (ht_float_t)32.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&v), (ht_float_t)(32.0 / 7.0), 1e-4);

    /* Weighted Welford test: y1 = 10 (w=2), y2 = 20 (w=2) -> W=4, mean=15, M2=100, s^2=100/3 */
    ht_var_stat_t wv;
    ht_var_init(&wv);
    ht_var_update(&wv, (ht_float_t)10.0, (ht_float_t)2.0);
    ht_var_update(&wv, (ht_float_t)20.0, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_EQ(wv.count, (ht_float_t)4.0);
    HT_ASSERT_FLOAT_NEAR(wv.mean, (ht_float_t)15.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(wv.m2, (ht_float_t)100.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&wv), (ht_float_t)(100.0 / 3.0), 1e-4);
}

static void test_var_stat_chan_combine_analytical(void) {
    /* Set A: {10.0, 20.0} -> nA = 2, meanA = 15.0, M2_A = 50.0 */
    ht_var_stat_t a;
    ht_var_init(&a);
    ht_var_update(&a, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_var_update(&a, (ht_float_t)20.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(a.count, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_NEAR(a.mean, (ht_float_t)15.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(a.m2, (ht_float_t)50.0, 1e-5);

    /* Set B: {30.0, 40.0} -> nB = 2, meanB = 35.0, M2_B = 50.0 */
    ht_var_stat_t b;
    ht_var_init(&b);
    ht_var_update(&b, (ht_float_t)30.0, (ht_float_t)1.0);
    ht_var_update(&b, (ht_float_t)40.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(b.count, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_NEAR(b.mean, (ht_float_t)35.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(b.m2, (ht_float_t)50.0, 1e-5);

    /* Combined C = A \cup B:
     * nC = 4, meanC = (30 + 70)/4 = 25.0
     * delta = 35 - 15 = 20
     * M2_C = 50 + 50 + 20^2 * (2 * 2 / 4) = 100 + 400 * 1 = 500.0
     * s^2 = 500 / 3 = 166.666667
     */
    ht_var_stat_t c = ht_var_combine(&a, &b);
    HT_ASSERT_FLOAT_EQ(c.count, (ht_float_t)4.0);
    HT_ASSERT_FLOAT_NEAR(c.mean, (ht_float_t)25.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(c.m2, (ht_float_t)500.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&c), (ht_float_t)(500.0 / 3.0), 1e-4);

    /* Verify exact match with single sequential Welford on {10, 20, 30, 40} */
    ht_var_stat_t seq;
    ht_var_init(&seq);
    ht_var_update(&seq, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_var_update(&seq, (ht_float_t)20.0, (ht_float_t)1.0);
    ht_var_update(&seq, (ht_float_t)30.0, (ht_float_t)1.0);
    ht_var_update(&seq, (ht_float_t)40.0, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_EQ(c.count, seq.count);
    HT_ASSERT_FLOAT_NEAR(c.mean, seq.mean, 1e-5);
    HT_ASSERT_FLOAT_NEAR(c.m2, seq.m2, 1e-4);

    /* Edge cases: combine with empty / NULL */
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
    HT_ASSERT_FLOAT_EQ(c_null.count, (ht_float_t)0.0);
}

static void test_var_stat_chan_subtract_analytical(void) {
    /* Construct total set C = {10, 20, 30, 40} */
    ht_var_stat_t a, b;
    ht_var_init(&a);
    ht_var_init(&b);
    ht_var_update(&a, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_var_update(&a, (ht_float_t)20.0, (ht_float_t)1.0);
    ht_var_update(&b, (ht_float_t)30.0, (ht_float_t)1.0);
    ht_var_update(&b, (ht_float_t)40.0, (ht_float_t)1.0);

    ht_var_stat_t c = ht_var_combine(&a, &b);

    /* Subtract A from C: R = C \setminus A -> must exactly match B */
    ht_var_stat_t r = ht_var_subtract(&c, &a);
    HT_ASSERT_FLOAT_EQ(r.count, b.count);
    HT_ASSERT_FLOAT_NEAR(r.mean, b.mean, 1e-5);
    HT_ASSERT_FLOAT_NEAR(r.m2, b.m2, 1e-4);
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&r), ht_var_value(&b), 1e-4);

    /* Subtract B from C: R = C \setminus B -> must exactly match A */
    ht_var_stat_t r2 = ht_var_subtract(&c, &b);
    HT_ASSERT_FLOAT_EQ(r2.count, a.count);
    HT_ASSERT_FLOAT_NEAR(r2.mean, a.mean, 1e-5);
    HT_ASSERT_FLOAT_NEAR(r2.m2, a.m2, 1e-4);

    /* Edge cases: subtract empty part -> returns total */
    ht_var_stat_t empty;
    ht_var_init(&empty);
    ht_var_stat_t r_empty = ht_var_subtract(&c, &empty);
    HT_ASSERT_FLOAT_EQ(r_empty.count, c.count);
    HT_ASSERT_FLOAT_NEAR(r_empty.mean, c.mean, 1e-5);
    HT_ASSERT_FLOAT_NEAR(r_empty.m2, c.m2, 1e-4);

    /* Edge cases: subtract total from total -> returns zeroed */
    ht_var_stat_t r_all = ht_var_subtract(&c, &c);
    HT_ASSERT_FLOAT_EQ(r_all.count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(r_all.mean, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(r_all.m2, (ht_float_t)0.0);

    /* Edge cases: NULL total */
    ht_var_stat_t r_null = ht_var_subtract(NULL, &a);
    HT_ASSERT_FLOAT_EQ(r_null.count, (ht_float_t)0.0);
}

static void test_reg_numeric_obs_ordered_slots_and_merging(void) {
    ht_reg_numeric_obs_t obs;
    ht_reg_numeric_init(&obs);

    /* Verify all slots zeroed */
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        HT_ASSERT_FLOAT_EQ(obs.slots[k].x_mean, (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(obs.slots[k].y_stats.count, (ht_float_t)0.0);
    }

    /* Insert 5 samples: x = 10, 30, 20, 50, 40 (unordered) */
    ht_reg_numeric_update(&obs, (ht_float_t)10.0, (ht_float_t)1.0, (ht_float_t)1.0);
    ht_reg_numeric_update(&obs, (ht_float_t)30.0, (ht_float_t)3.0, (ht_float_t)1.0);
    ht_reg_numeric_update(&obs, (ht_float_t)20.0, (ht_float_t)2.0, (ht_float_t)1.0);
    ht_reg_numeric_update(&obs, (ht_float_t)50.0, (ht_float_t)5.0, (ht_float_t)1.0);
    ht_reg_numeric_update(&obs, (ht_float_t)40.0, (ht_float_t)4.0, (ht_float_t)1.0);

    /* Verify slots are strictly sorted by x_mean: 10, 20, 30, 40, 50 */
    HT_ASSERT_FLOAT_NEAR(obs.slots[0].x_mean, (ht_float_t)10.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.slots[1].x_mean, (ht_float_t)20.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.slots[2].x_mean, (ht_float_t)30.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.slots[3].x_mean, (ht_float_t)40.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.slots[4].x_mean, (ht_float_t)50.0, 1e-5);
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        HT_ASSERT_FLOAT_EQ(obs.slots[k].y_stats.count, (ht_float_t)1.0);
    }

    /* Insert 6th sample: x = 21.0. All 5 slots full.
     * Adjacent distances:
     * (10, 20) = 10
     * (20, 30) = 10
     * (30, 40) = 10
     * (40, 50) = 10
     * First pair with min_dist (10) is index 0: slots (10, 20) are fused into x_mean = 15.0, count = 2.
     * Then 21.0 is inserted in sorted order!
     */
    ht_reg_numeric_update(&obs, (ht_float_t)21.0, (ht_float_t)2.1, (ht_float_t)1.0);

    /* Verify slots remain strictly sorted */
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS - 1; ++k) {
        HT_ASSERT(obs.slots[k].x_mean <= obs.slots[k + 1].x_mean);
    }

    /* Stream 100 pseudo-random continuous samples */
    uint32_t lcg = 12345;
    ht_float_t sum_w = (ht_float_t)0.0;
    ht_float_t sum_xw = (ht_float_t)0.0;

    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        sum_w += obs.slots[k].y_stats.count;
        sum_xw += obs.slots[k].y_stats.count * obs.slots[k].x_mean;
    }

    for (int i = 0; i < 100; ++i) {
        lcg = lcg * 1664525U + 1013904223U;
        ht_float_t x_in = (ht_float_t)(lcg % 1000) * (ht_float_t)0.1; /* [0.0 .. 99.9] */
        ht_float_t y_in = (ht_float_t)(x_in * 2.5 - 10.0);
        ht_float_t w_in = (ht_float_t)1.0;

        sum_w += w_in;
        sum_xw += x_in * w_in;

        ht_reg_numeric_update(&obs, x_in, y_in, w_in);

        /* Invariant: slots MUST remain strictly ordered by x_mean after every insertion */
        for (uint8_t k = 0; k < HT_REG_NUM_SLOTS - 1; ++k) {
            HT_ASSERT(obs.slots[k].x_mean <= obs.slots[k + 1].x_mean);
            HT_ASSERT(obs.slots[k].y_stats.count > (ht_float_t)0.0);
        }
    }

    /* Conservation of total weight across all slots */
    ht_float_t total_slot_w = (ht_float_t)0.0;
    ht_float_t total_slot_xw = (ht_float_t)0.0;
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        total_slot_w += obs.slots[k].y_stats.count;
        total_slot_xw += obs.slots[k].y_stats.count * obs.slots[k].x_mean;
    }
    HT_ASSERT_FLOAT_NEAR(total_slot_w, sum_w, 1e-4);
    HT_ASSERT_FLOAT_NEAR(total_slot_xw, sum_xw, 1e-2);

    /* Edge cases: NaN, infinite, weight <= 0 must be ignored */
    ht_float_t nan_val = (ht_float_t)0.0 / (ht_float_t)0.0;
    ht_reg_numeric_update(&obs, nan_val, (ht_float_t)1.0, (ht_float_t)1.0);
    ht_reg_numeric_update(&obs, (ht_float_t)1.0, nan_val, (ht_float_t)1.0);
    ht_reg_numeric_update(&obs, (ht_float_t)1.0, (ht_float_t)1.0, (ht_float_t)0.0);
    ht_reg_numeric_update(&obs, (ht_float_t)1.0, (ht_float_t)1.0, (ht_float_t)-5.0);
    ht_reg_numeric_update(NULL, (ht_float_t)1.0, (ht_float_t)1.0, (ht_float_t)1.0);

    /* Weight unchanged */
    ht_float_t after_edge_w = (ht_float_t)0.0;
    for (uint8_t k = 0; k < HT_REG_NUM_SLOTS; ++k) {
        after_edge_w += obs.slots[k].y_stats.count;
    }
    HT_ASSERT_FLOAT_NEAR(after_edge_w, sum_w, 1e-4);
}

static void test_reg_categorical_obs_basic_and_edge_cases(void) {
    ht_reg_categorical_obs_t cat;
    ht_reg_categorical_init(&cat);

    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
        HT_ASSERT_FLOAT_EQ(cat.cat_stats[v].count, (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(cat.cat_stats[v].mean, (ht_float_t)0.0);
        HT_ASSERT_FLOAT_EQ(cat.cat_stats[v].m2, (ht_float_t)0.0);
    }

    /* Category 0: targets {10, 20} -> n = 2, mean = 15, M2 = 50 */
    ht_reg_categorical_update(&cat, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_reg_categorical_update(&cat, 0, (ht_float_t)20.0, (ht_float_t)1.0);

    /* Category 2: targets {5, 5, 5} -> n = 3, mean = 5, M2 = 0 */
    ht_reg_categorical_update(&cat, 2, (ht_float_t)5.0, (ht_float_t)1.0);
    ht_reg_categorical_update(&cat, 2, (ht_float_t)5.0, (ht_float_t)1.0);
    ht_reg_categorical_update(&cat, 2, (ht_float_t)5.0, (ht_float_t)1.0);

    HT_ASSERT_FLOAT_EQ(cat.cat_stats[0].count, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[0].mean, (ht_float_t)15.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[0].m2, (ht_float_t)50.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&cat.cat_stats[0]), (ht_float_t)50.0, 1e-4);

    HT_ASSERT_FLOAT_EQ(cat.cat_stats[2].count, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[2].mean, (ht_float_t)5.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(cat.cat_stats[2].m2, (ht_float_t)0.0, 1e-5);
    HT_ASSERT_FLOAT_EQ(ht_var_value(&cat.cat_stats[2]), (ht_float_t)0.0);

    /* Category 1, 3, 4 unobserved */
    HT_ASSERT_FLOAT_EQ(cat.cat_stats[1].count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(cat.cat_stats[3].count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(cat.cat_stats[4].count, (ht_float_t)0.0);

    /* Edge cases: negative category, >= HT_MAX_CATEGORIES, NaN target, non-positive weight, NULL */
    ht_reg_categorical_update(&cat, -1, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_reg_categorical_update(&cat, HT_MAX_CATEGORIES, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_float_t nan_val = (ht_float_t)0.0 / (ht_float_t)0.0;
    ht_reg_categorical_update(&cat, 0, nan_val, (ht_float_t)1.0);
    ht_reg_categorical_update(&cat, 0, (ht_float_t)10.0, (ht_float_t)0.0);
    ht_reg_categorical_update(&cat, 0, (ht_float_t)10.0, (ht_float_t)-2.0);
    ht_reg_categorical_update(NULL, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_reg_categorical_init(NULL);

    HT_ASSERT_FLOAT_EQ(cat.cat_stats[0].count, (ht_float_t)2.0);
}

static void test_reg_observers_in_leaf_stats_pool(void) {
    ht_memory_init(&s_observer_test_pool);

    uint16_t stats_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);

    ht_leaf_stats_t *stats = ht_stats_get(&s_observer_test_pool, stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    /* Update numeric regression observer at feature 0 */
    ht_reg_numeric_update(&stats->observers[0].reg_num, (ht_float_t)5.0, (ht_float_t)12.0, (ht_float_t)1.0);
    ht_reg_numeric_update(&stats->observers[0].reg_num, (ht_float_t)15.0, (ht_float_t)24.0, (ht_float_t)1.0);

    HT_ASSERT_FLOAT_NEAR(stats->observers[0].reg_num.slots[0].x_mean, (ht_float_t)5.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(stats->observers[0].reg_num.slots[1].x_mean, (ht_float_t)15.0, 1e-5);

    /* Update categorical regression observer at feature 1 */
    ht_reg_categorical_update(&stats->observers[1].reg_cat, 3, (ht_float_t)42.0, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_EQ(stats->observers[1].reg_cat.cat_stats[3].count, (ht_float_t)2.0);
    HT_ASSERT_FLOAT_NEAR(stats->observers[1].reg_cat.cat_stats[3].mean, (ht_float_t)42.0, 1e-5);

    /* Free and reallocate: must return zeroed */
    ht_stats_free(&s_observer_test_pool, stats_idx);
    HT_ASSERT_EQ(ht_stats_count(&s_observer_test_pool), 0);

    uint16_t realloc_idx = ht_stats_alloc(&s_observer_test_pool);
    HT_ASSERT_EQ(realloc_idx, stats_idx);

    ht_leaf_stats_t *re_stats = ht_stats_get(&s_observer_test_pool, realloc_idx);
    HT_ASSERT_NOT_NULL(re_stats);
    HT_ASSERT_FLOAT_EQ(re_stats->observers[0].reg_num.slots[0].y_stats.count, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(re_stats->observers[1].reg_cat.cat_stats[3].count, (ht_float_t)0.0);

    ht_stats_free(&s_observer_test_pool, realloc_idx);
}

/* ========================================================================= */
/* --- Level 1 & 2: Histogram Non-Parametric Observer Tests ---------------- */
/* ========================================================================= */

static void test_hist_obs_basic_and_merging(void) {
    ht_hist_obs_t obs;
    ht_hist_init(&obs);

    /* Insert 4 distinct values */
    ht_hist_update(&obs, 0, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_hist_update(&obs, 0, (ht_float_t)20.0, (ht_float_t)1.0);
    ht_hist_update(&obs, 0, (ht_float_t)30.0, (ht_float_t)1.0);
    ht_hist_update(&obs, 0, (ht_float_t)40.0, (ht_float_t)1.0);

    HT_ASSERT_FLOAT_NEAR(obs.bins[0].center, (ht_float_t)10.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.bins[1].center, (ht_float_t)20.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.bins[2].center, (ht_float_t)30.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.bins[3].center, (ht_float_t)40.0, 1e-5);

    /* Insert 5th value 11.0: distance between 10.0 and 11.0 is 1.0 (minimum).
     * Bins 10.0 and 11.0 merge into (10.0*1 + 11.0*1)/2 = 10.5 with count = 2.0 */
    ht_hist_update(&obs, 0, (ht_float_t)11.0, (ht_float_t)1.0);

    HT_ASSERT_FLOAT_NEAR(obs.bins[0].center, (ht_float_t)10.5, 1e-4);
    HT_ASSERT_FLOAT_NEAR(obs.bins[0].counts[0], (ht_float_t)2.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(obs.bins[1].center, (ht_float_t)20.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(obs.bins[2].center, (ht_float_t)30.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(obs.bins[3].center, (ht_float_t)40.0, 1e-4);

    /* Total class 0 weight must be 5.0 */
    ht_float_t total = (ht_float_t)0.0;
    for (uint8_t i = 0; i < HT_HIST_MAX_BINS; ++i) {
        total += obs.bins[i].counts[0];
    }
    HT_ASSERT_FLOAT_NEAR(total, (ht_float_t)5.0, 1e-5);
}

static void test_hist_obs_split_dists_and_proba(void) {
    ht_hist_obs_t obs;
    ht_hist_init(&obs);

    /* Class 0 at low values: 1.0, 2.0 */
    ht_hist_update(&obs, 0, (ht_float_t)1.0, (ht_float_t)1.0);
    ht_hist_update(&obs, 0, (ht_float_t)2.0, (ht_float_t)1.0);

    /* Class 1 at high values: 10.0, 11.0 */
    ht_hist_update(&obs, 1, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_hist_update(&obs, 1, (ht_float_t)11.0, (ht_float_t)1.0);

    ht_float_t splits[HT_GAUSSIAN_SPLITS];
    uint8_t n_splits = ht_hist_split_suggestions(&obs, 2, splits);
    HT_ASSERT(n_splits >= 1);

    /* Evaluate binary split at threshold 5.0 (between clusters 1..2 and 10..11) */
    ht_float_t lhs[HT_MAX_CLASSES];
    ht_float_t rhs[HT_MAX_CLASSES];
    ht_hist_binary_split_dists(&obs, 2, (ht_float_t)5.0, lhs, rhs);

    /* Class 0 all on left, Class 1 all on right */
    HT_ASSERT_FLOAT_NEAR(lhs[0], (ht_float_t)2.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(lhs[1], (ht_float_t)0.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(rhs[0], (ht_float_t)0.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(rhs[1], (ht_float_t)2.0, 1e-4);

    /* Conditional probabilities */
    ht_float_t p0 = ht_hist_cond_proba(&obs, 0, (ht_float_t)1.5);
    ht_float_t p1 = ht_hist_cond_proba(&obs, 1, (ht_float_t)1.5);
    HT_ASSERT(p0 > (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(p1, (ht_float_t)0.0);
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

    /* Nível 2: Integração com ht_memory slab pool */
    HT_RUN_TEST(test_observers_in_leaf_stats_pool);
    HT_RUN_TEST(test_reg_observers_in_leaf_stats_pool);
}
