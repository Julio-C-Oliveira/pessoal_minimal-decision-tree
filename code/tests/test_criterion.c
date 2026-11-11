/**
 * @file test_criterion.c
 * @brief Level 1 & Level 2 tests for split criteria (Gini, InfoGain), Hoeffding bound, and candidate search.
 */

#include "test_harness.h"
#include "ht_criterion.h"
#include "ht_memory.h"
#include "ht_observer.h"

/* ========================================================================= */
/* --- Level 1: Analytical Gini Impurity Tests ----------------------------- */
/* ========================================================================= */

static void test_gini_impurity_analytical(void) {
    /* 1. Pure distribution [100, 0, 0, 0] -> Gini = 0.0 */
    ht_float_t pure_dist[HT_MAX_CLASSES] = { (ht_float_t)100.0, 0, 0, 0 };
    ht_float_t g_pure = ht_compute_gini(pure_dist, 2, (ht_float_t)100.0);
    HT_ASSERT_FLOAT_NEAR(g_pure, (ht_float_t)0.0, 1e-6);

    /* 2. Equal binary distribution [50, 50, 0, 0] -> Gini = 1 - (0.5^2 + 0.5^2) = 0.5 */
    ht_float_t eq2_dist[HT_MAX_CLASSES] = { (ht_float_t)50.0, (ht_float_t)50.0, 0, 0 };
    ht_float_t g_eq2 = ht_compute_gini(eq2_dist, 2, (ht_float_t)100.0);
    HT_ASSERT_FLOAT_NEAR(g_eq2, (ht_float_t)0.5, 1e-6);

    /* 3. Equal quad distribution [25, 25, 25, 25] -> Gini = 1 - 4 * 0.25^2 = 0.75 */
    ht_float_t eq4_dist[HT_MAX_CLASSES] = {
        (ht_float_t)25.0, (ht_float_t)25.0, (ht_float_t)25.0, (ht_float_t)25.0
    };
    ht_float_t g_eq4 = ht_compute_gini(eq4_dist, 4, (ht_float_t)100.0);
    HT_ASSERT_FLOAT_NEAR(g_eq4, (ht_float_t)0.75, 1e-6);

    /* 4. Skewed distribution [90, 10] -> Gini = 1 - (0.9^2 + 0.1^2) = 1 - 0.82 = 0.18 */
    ht_float_t skew_dist[HT_MAX_CLASSES] = { (ht_float_t)90.0, (ht_float_t)10.0, 0, 0 };
    ht_float_t g_skew = ht_compute_gini(skew_dist, 2, (ht_float_t)100.0);
    HT_ASSERT_FLOAT_NEAR(g_skew, (ht_float_t)0.18, 1e-6);

    /* 5. Empty distribution [0, 0] -> Gini = 1.0 */
    ht_float_t empty_dist[HT_MAX_CLASSES] = { 0, 0, 0, 0 };
    ht_float_t g_empty = ht_compute_gini(empty_dist, 2, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_NEAR(g_empty, (ht_float_t)1.0, 1e-6);
}

static void test_gini_merit_and_min_branch(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);
    HT_ASSERT_NOT_NULL(crit);

    ht_float_t pre_dist[HT_MAX_CLASSES] = { (ht_float_t)50.0, (ht_float_t)50.0, 0, 0 };

    /* 1. Perfect split: left=[50, 0], right=[0, 50] -> merit = 1.0 - 0.0 = 1.0 */
    ht_float_t lhs_perf[HT_MAX_CLASSES] = { (ht_float_t)50.0, 0, 0, 0 };
    ht_float_t rhs_perf[HT_MAX_CLASSES] = { 0, (ht_float_t)50.0, 0, 0 };
    ht_float_t m_perf = crit->merit_of_split(pre_dist, lhs_perf, rhs_perf, 2, (ht_float_t)0.01);
    HT_ASSERT_FLOAT_NEAR(m_perf, (ht_float_t)1.0, 1e-6);

    /* 2. Useless split: left=[25, 25], right=[25, 25] -> merit = 1.0 - 0.5 = 0.5 */
    ht_float_t lhs_use[HT_MAX_CLASSES] = { (ht_float_t)25.0, (ht_float_t)25.0, 0, 0 };
    ht_float_t rhs_use[HT_MAX_CLASSES] = { (ht_float_t)25.0, (ht_float_t)25.0, 0, 0 };
    ht_float_t m_use = crit->merit_of_split(pre_dist, lhs_use, rhs_use, 2, (ht_float_t)0.01);
    HT_ASSERT_FLOAT_NEAR(m_use, (ht_float_t)0.5, 1e-6);

    /* 3. Range of merit is always 1.0 */
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(pre_dist, 2), (ht_float_t)1.0, 1e-6);

    /* 4. min_branch_fraction test: min_frac = 0.05, total = 100 */
    /* lhs = 3 samples (3%), rhs = 97 samples (97%) -> rejected (-HT_INFINITY) */
    ht_float_t lhs_small[HT_MAX_CLASSES] = { (ht_float_t)3.0, 0, 0, 0 };
    ht_float_t rhs_large[HT_MAX_CLASSES] = { (ht_float_t)47.0, (ht_float_t)50.0, 0, 0 };
    ht_float_t m_rej = crit->merit_of_split(pre_dist, lhs_small, rhs_large, 2, (ht_float_t)0.05);
    HT_ASSERT_EQ(m_rej, -HT_INFINITY);

    /* Exact boundary: lhs = 5.0 (5.0%) with min_frac = 0.05 -> rejected because frac <= min_frac */
    ht_float_t lhs_exact[HT_MAX_CLASSES] = { (ht_float_t)5.0, 0, 0, 0 };
    ht_float_t rhs_exact[HT_MAX_CLASSES] = { (ht_float_t)45.0, (ht_float_t)50.0, 0, 0 };
    ht_float_t m_bnd = crit->merit_of_split(pre_dist, lhs_exact, rhs_exact, 2, (ht_float_t)0.05);
    HT_ASSERT_EQ(m_bnd, -HT_INFINITY);

    /* Valid: lhs = 6.0 (6.0%) > 0.05 -> accepted */
    ht_float_t lhs_valid[HT_MAX_CLASSES] = { (ht_float_t)6.0, 0, 0, 0 };
    ht_float_t rhs_valid[HT_MAX_CLASSES] = { (ht_float_t)44.0, (ht_float_t)50.0, 0, 0 };
    ht_float_t m_val = crit->merit_of_split(pre_dist, lhs_valid, rhs_valid, 2, (ht_float_t)0.05);
    HT_ASSERT(m_val > -HT_INFINITY);

    /* Total weight zero -> rejected */
    ht_float_t zero_dist[HT_MAX_CLASSES] = { 0, 0, 0, 0 };
    HT_ASSERT_EQ(crit->merit_of_split(pre_dist, zero_dist, zero_dist, 2, (ht_float_t)0.01), -HT_INFINITY);
}

/* ========================================================================= */
/* --- Level 1: Analytical Information Gain Tests -------------------------- */
/* ========================================================================= */

static void test_entropy_analytical(void) {
    /* 1. Pure distribution [100, 0] -> H = 0.0 */
    ht_float_t pure_dist[HT_MAX_CLASSES] = { (ht_float_t)100.0, 0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(pure_dist, 2), (ht_float_t)0.0, 1e-6);

    /* 2. Equal binary distribution [50, 50] -> H = 1.0 */
    ht_float_t eq2_dist[HT_MAX_CLASSES] = { (ht_float_t)50.0, (ht_float_t)50.0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(eq2_dist, 2), (ht_float_t)1.0, 1e-6);

    /* 3. Equal quad distribution [25, 25, 25, 25] -> H = log2(4) = 2.0 */
    ht_float_t eq4_dist[HT_MAX_CLASSES] = {
        (ht_float_t)25.0, (ht_float_t)25.0, (ht_float_t)25.0, (ht_float_t)25.0
    };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(eq4_dist, 4), (ht_float_t)2.0, 1e-6);

    /* 4. Skewed [80, 20] -> H = -(0.8 log2(0.8) + 0.2 log2(0.2)) ~= 0.72192809 */
    ht_float_t skew_dist[HT_MAX_CLASSES] = { (ht_float_t)80.0, (ht_float_t)20.0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(skew_dist, 2), (ht_float_t)0.72192809, 1e-5);

    /* 5. Empty distribution -> H = 0.0 */
    ht_float_t empty_dist[HT_MAX_CLASSES] = { 0, 0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(empty_dist, 2), (ht_float_t)0.0, 1e-6);
}

static void test_info_gain_merit_and_range(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_INFO_GAIN);
    HT_ASSERT_NOT_NULL(crit);

    ht_float_t pre_dist[HT_MAX_CLASSES] = { (ht_float_t)50.0, (ht_float_t)50.0, 0, 0 };

    /* 1. Perfect split: left=[50, 0], right=[0, 50] -> merit = 1.0 - 0.0 = 1.0 */
    ht_float_t lhs_perf[HT_MAX_CLASSES] = { (ht_float_t)50.0, 0, 0, 0 };
    ht_float_t rhs_perf[HT_MAX_CLASSES] = { 0, (ht_float_t)50.0, 0, 0 };
    ht_float_t m_perf = crit->merit_of_split(pre_dist, lhs_perf, rhs_perf, 2, (ht_float_t)0.01);
    HT_ASSERT_FLOAT_NEAR(m_perf, (ht_float_t)1.0, 1e-6);

    /* 2. Useless split: left=[25, 25], right=[25, 25] -> merit = 1.0 - 1.0 = 0.0 */
    ht_float_t lhs_use[HT_MAX_CLASSES] = { (ht_float_t)25.0, (ht_float_t)25.0, 0, 0 };
    ht_float_t rhs_use[HT_MAX_CLASSES] = { (ht_float_t)25.0, (ht_float_t)25.0, 0, 0 };
    ht_float_t m_use = crit->merit_of_split(pre_dist, lhs_use, rhs_use, 2, (ht_float_t)0.01);
    HT_ASSERT_FLOAT_NEAR(m_use, (ht_float_t)0.0, 1e-6);

    /* 3. Range of merit: log2(max(C_obs, 2)) */
    /* 1 class observed -> max(1, 2) = 2 -> log2(2) = 1.0 */
    ht_float_t dist_1c[HT_MAX_CLASSES] = { (ht_float_t)100.0, 0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dist_1c, 4), (ht_float_t)1.0, 1e-6);

    /* 2 classes observed -> max(2, 2) = 2 -> log2(2) = 1.0 */
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(pre_dist, 4), (ht_float_t)1.0, 1e-6);

    /* 3 classes observed -> max(3, 2) = 3 -> log2(3) ~= 1.5849625 */
    ht_float_t dist_3c[HT_MAX_CLASSES] = { (ht_float_t)30.0, (ht_float_t)30.0, (ht_float_t)40.0, 0 };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dist_3c, 4), (ht_float_t)1.5849625, 1e-5);

    /* 4 classes observed -> max(4, 2) = 4 -> log2(4) = 2.0 */
    ht_float_t dist_4c[HT_MAX_CLASSES] = {
        (ht_float_t)25.0, (ht_float_t)25.0, (ht_float_t)25.0, (ht_float_t)25.0
    };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dist_4c, 4), (ht_float_t)2.0, 1e-6);

    /* 4. min_branch_fraction test */
    ht_float_t lhs_small[HT_MAX_CLASSES] = { (ht_float_t)2.0, 0, 0, 0 };
    ht_float_t rhs_large[HT_MAX_CLASSES] = { (ht_float_t)48.0, (ht_float_t)50.0, 0, 0 };
    HT_ASSERT_EQ(crit->merit_of_split(pre_dist, lhs_small, rhs_large, 2, (ht_float_t)0.05), -HT_INFINITY);
}

/* ========================================================================= */
/* --- Level 1: Analytical Hoeffding Bound Tests --------------------------- */
/* ========================================================================= */

static void test_hoeffding_bound_analytical(void) {
    /* epsilon = sqrt((R^2 * ln(1 / delta)) / (2 * n)) */
    /* For R = 1.0, delta = 1e-7, n = 200:
     * ln(1e7) = 16.11809565095832
     * 2 * n = 400
     * eps = sqrt(16.11809565 / 400) = sqrt(0.040295239) ~= 0.20073674 */
    ht_float_t eps_200 = ht_hoeffding_bound((ht_float_t)1.0, (ht_float_t)1e-7, (ht_float_t)200.0);
    HT_ASSERT_FLOAT_NEAR(eps_200, (ht_float_t)0.20073674, 1e-4);

    /* For R = 2.0, eps should double: ~= 0.40147348 */
    ht_float_t eps_r2 = ht_hoeffding_bound((ht_float_t)2.0, (ht_float_t)1e-7, (ht_float_t)200.0);
    HT_ASSERT_FLOAT_NEAR(eps_r2, (ht_float_t)0.40147348, 1e-4);

    /* Monotonicity: for n = 2000, eps must be strictly smaller */
    ht_float_t eps_2000 = ht_hoeffding_bound((ht_float_t)1.0, (ht_float_t)1e-7, (ht_float_t)2000.0);
    HT_ASSERT(eps_2000 < eps_200);
    /* For n = 2000: sqrt(16.11809565 / 4000) = sqrt(0.004029524) ~= 0.06347853 */
    HT_ASSERT_FLOAT_NEAR(eps_2000, (ht_float_t)0.06347853, 1e-4);

    /* Edge cases */
    HT_ASSERT_EQ(ht_hoeffding_bound((ht_float_t)1.0, (ht_float_t)1e-7, (ht_float_t)0.0), HT_INFINITY);
    HT_ASSERT_EQ(ht_hoeffding_bound((ht_float_t)1.0, (ht_float_t)1e-7, (ht_float_t)-5.0), HT_INFINITY);
    HT_ASSERT_EQ(ht_hoeffding_bound((ht_float_t)1.0, (ht_float_t)0.0, (ht_float_t)100.0), HT_INFINITY);
    HT_ASSERT_FLOAT_EQ(ht_hoeffding_bound((ht_float_t)1.0, (ht_float_t)1.0, (ht_float_t)100.0), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_hoeffding_bound((ht_float_t)1.0, (ht_float_t)1.5, (ht_float_t)100.0), (ht_float_t)0.0);
}

/* ========================================================================= */
/* --- Level 1: Observer Evaluators Tests ---------------------------------- */
/* ========================================================================= */

static void test_evaluate_numeric_observer(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);
    ht_gaussian_obs_t obs;
    ht_gaussian_init(&obs);

    /* Populate Class 0: values in [0.0, 4.0] (mean 2.0) */
    for (int i = 0; i <= 20; i++) {
        ht_gaussian_update(&obs, 0, (ht_float_t)(i * 0.2), (ht_float_t)1.0);
    }
    /* Populate Class 1: values in [6.0, 10.0] (mean 8.0) */
    for (int i = 0; i <= 20; i++) {
        ht_gaussian_update(&obs, 1, (ht_float_t)(6.0 + i * 0.2), (ht_float_t)1.0);
    }

    ht_float_t pre_dist[HT_MAX_CLASSES] = { (ht_float_t)21.0, (ht_float_t)21.0, 0, 0 };
    ht_split_candidate_t cand;

    ht_evaluate_numeric_observer(&obs, pre_dist, 2, 0, crit, (ht_float_t)0.01, &cand);

    HT_ASSERT_EQ(cand.feature_idx, 0);
    HT_ASSERT_EQ(cand.is_categorical, 0);
    HT_ASSERT(cand.merit > (ht_float_t)0.8); /* High merit because well-separated */
    /* Threshold should be in between [4.0, 6.0] */
    HT_ASSERT(cand.split_val.threshold >= (ht_float_t)3.5 && cand.split_val.threshold <= (ht_float_t)6.5);

    /* Test constant feature: all samples equal to 5.0 */
    ht_gaussian_obs_t const_obs;
    ht_gaussian_init(&const_obs);
    for (int i = 0; i < 20; i++) {
        ht_gaussian_update(&const_obs, 0, (ht_float_t)5.0, (ht_float_t)1.0);
        ht_gaussian_update(&const_obs, 1, (ht_float_t)5.0, (ht_float_t)1.0);
    }
    ht_split_candidate_t const_cand;
    ht_evaluate_numeric_observer(&const_obs, pre_dist, 2, 1, crit, (ht_float_t)0.01, &const_cand);
    HT_ASSERT_EQ(const_cand.merit, -HT_INFINITY);
    HT_ASSERT_EQ(const_cand.feature_idx, HT_NO_FEATURE);
}

static void test_evaluate_categorical_observer(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);
    ht_categorical_obs_t cat_obs;
    ht_categorical_init(&cat_obs);

    /* Class 0 always category 1 (count 50) */
    ht_categorical_update(&cat_obs, 0, 1, (ht_float_t)50.0);
    /* Class 1 always category 3 (count 50) */
    ht_categorical_update(&cat_obs, 1, 3, (ht_float_t)50.0);

    ht_float_t pre_dist[HT_MAX_CLASSES] = { (ht_float_t)50.0, (ht_float_t)50.0, 0, 0 };
    ht_split_candidate_t cand;

    ht_evaluate_categorical_observer(&cat_obs, pre_dist, 2, 2, crit, (ht_float_t)0.01, &cand);

    HT_ASSERT_EQ(cand.feature_idx, 2);
    HT_ASSERT_EQ(cand.is_categorical, 1);
    HT_ASSERT_FLOAT_NEAR(cand.merit, (ht_float_t)1.0, 1e-6); /* Perfect separation */
    /* Either category 1 or 3 isolates one class completely */
    HT_ASSERT(cand.split_val.cat_val == 1 || cand.split_val.cat_val == 3);

    /* Test empty categorical observer */
    ht_categorical_obs_t empty_cat;
    ht_categorical_init(&empty_cat);
    ht_split_candidate_t empty_cand;
    ht_evaluate_categorical_observer(&empty_cat, pre_dist, 2, 3, crit, (ht_float_t)0.01, &empty_cand);
    HT_ASSERT_EQ(empty_cand.merit, -HT_INFINITY);
    HT_ASSERT_EQ(empty_cand.feature_idx, HT_NO_FEATURE);
}

/* ========================================================================= */
/* --- Level 2: Multi-Attribute Candidate Search & Tie-Breaking ----------- */
/* ========================================================================= */

static void test_find_best_split_candidates_and_tie_breaking(void) {
    static ht_memory_pool_t pool;
    ht_memory_init(&pool);

    uint16_t stats_idx = ht_stats_alloc(&pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);
    ht_leaf_stats_t *stats = &pool.stats_pool[stats_idx];

    /* Schema: 3 features
     * feat 0: numeric (poor/uninformative, class 0 and 1 overlapping)
     * feat 1: numeric (informative, clean separation)
     * feat 2: categorical (informative, clean separation)
     */
    ht_feature_type_t types[HT_MAX_FEATURES];
    types[0] = HT_FEAT_NUMERIC;
    types[1] = HT_FEAT_NUMERIC;
    types[2] = HT_FEAT_CATEGORICAL;

    /* Populate Feature 0: overlapping around 5.0 for both classes */
    for (int i = 0; i < 50; i++) {
        ht_gaussian_update(&stats->observers[0].num, 0, (ht_float_t)(4.0 + i * 0.04), (ht_float_t)1.0);
        ht_gaussian_update(&stats->observers[0].num, 1, (ht_float_t)(4.0 + i * 0.04), (ht_float_t)1.0);
    }

    /* Populate Feature 1: class 0 in [0..2], class 1 in [8..10] */
    for (int i = 0; i < 50; i++) {
        ht_gaussian_update(&stats->observers[1].num, 0, (ht_float_t)(i * 0.04), (ht_float_t)1.0);
        ht_gaussian_update(&stats->observers[1].num, 1, (ht_float_t)(8.0 + i * 0.04), (ht_float_t)1.0);
    }

    /* Populate Feature 2: class 0 in cat 0, class 1 in cat 1 */
    ht_categorical_update(&stats->observers[2].cat, 0, 0, (ht_float_t)50.0);
    ht_categorical_update(&stats->observers[2].cat, 1, 1, (ht_float_t)50.0);

    ht_float_t pre_dist[HT_MAX_CLASSES] = { (ht_float_t)50.0, (ht_float_t)50.0, 0, 0 };
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);

    ht_split_candidate_t best, second_best;
    ht_find_best_split_candidates(stats, pre_dist, types, 3, 2, crit,
                                  (ht_float_t)0.01, true, &best, &second_best);

    /* Both feature 1 and 2 achieve perfect separation (merit = 1.0).
     * Due to stable sort matching River's Timsort, feature 2 (higher index) is at best,
     * and feature 1 is at second_best! */
    HT_ASSERT_FLOAT_NEAR(best.merit, (ht_float_t)1.0, 1e-5);
    HT_ASSERT_FLOAT_NEAR(second_best.merit, (ht_float_t)1.0, 1e-5);
    HT_ASSERT_EQ(best.feature_idx, 2);
    HT_ASSERT_EQ(second_best.feature_idx, 1);

    /* Free allocated stats */
    ht_stats_free(&pool, stats_idx);
}

static void test_should_split_logic(void) {
    ht_split_candidate_t best, second_best;
    ht_split_candidate_init(&best);
    ht_split_candidate_init(&second_best);

    ht_float_t tau = (ht_float_t)0.05;

    /* 1. Invalid best candidate -> cannot split */
    HT_ASSERT(!ht_should_split(&best, &second_best, (ht_float_t)0.1, tau));

    /* 2. Best candidate valid, but second_best invalid -> split! */
    best.merit = (ht_float_t)0.8;
    best.feature_idx = 0;
    HT_ASSERT(ht_should_split(&best, &second_best, (ht_float_t)0.5, tau));

    /* 3. Both valid: diff = 0.8 - 0.5 = 0.3.
     * hoeffding_bound = 0.2 < diff -> split! */
    second_best.merit = (ht_float_t)0.5;
    second_best.feature_idx = 1;
    HT_ASSERT(ht_should_split(&best, &second_best, (ht_float_t)0.2, tau));

    /* 4. Both valid: diff = 0.3.
     * hoeffding_bound = 0.4 > diff AND hoeffding_bound >= tau -> do NOT split */
    HT_ASSERT(!ht_should_split(&best, &second_best, (ht_float_t)0.4, tau));

    /* 5. Both valid: diff = 0.01 (near tie).
     * hoeffding_bound = 0.04 < tau (0.05) -> tie-break split triggered! */
    best.merit = (ht_float_t)0.81;
    second_best.merit = (ht_float_t)0.80;
    HT_ASSERT(ht_should_split(&best, &second_best, (ht_float_t)0.04, tau));

    /* 6. All candidates invalid (-HT_INFINITY) -> cannot split */
    best.merit = -HT_INFINITY;
    best.feature_idx = HT_NO_FEATURE;
    second_best.merit = -HT_INFINITY;
    second_best.feature_idx = HT_NO_FEATURE;
    HT_ASSERT(!ht_should_split(&best, &second_best, (ht_float_t)0.01, tau));
}

static void test_criterion_edge_cases_and_null(void) {
    /* Invalid criterion enum */
    HT_ASSERT_NULL(ht_criterion_get_vtable((ht_criterion_type_t)999));

    /* NULL pointers to functions should not crash */
    ht_split_candidate_t cand;
    ht_evaluate_numeric_observer(NULL, NULL, 0, 0, NULL, (ht_float_t)0.01, &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_evaluate_categorical_observer(NULL, NULL, 0, 0, NULL, (ht_float_t)0.01, &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_split_candidate_t best, second;
    ht_find_best_split_candidates(NULL, NULL, NULL, 0, 0, NULL, (ht_float_t)0.01, true, &best, &second);
    HT_ASSERT_EQ(best.merit, -HT_INFINITY);
    HT_ASSERT_EQ(second.merit, -HT_INFINITY);

    HT_ASSERT(!ht_should_split(NULL, NULL, (ht_float_t)0.1, (ht_float_t)0.05));
}

/* ========================================================================= */
/* --- Level 1 & 2: Regression Variance Reduction Split Tests -------------- */
/* ========================================================================= */

static void test_vr_analytical_and_filters(void) {
    /* 1. Analytical test: 10 samples (five 2.0s, five 8.0s)
     * Mean = 5.0, M2 = 5 * (-3)^2 + 5 * (3)^2 = 45 + 45 = 90.0
     * Var(D_pre) = 90.0 / 9 = 10.0 */
    ht_var_stat_t pre_var;
    ht_var_init(&pre_var);
    for (int i = 0; i < 5; i++) {
        ht_var_update(&pre_var, (ht_float_t)2.0, (ht_float_t)1.0);
        ht_var_update(&pre_var, (ht_float_t)8.0, (ht_float_t)1.0);
    }
    HT_ASSERT_FLOAT_NEAR(pre_var.count, (ht_float_t)10.0, 1e-6);
    HT_ASSERT_FLOAT_NEAR(pre_var.mean, (ht_float_t)5.0, 1e-6);
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&pre_var), (ht_float_t)10.0, 1e-6);

    /* Perfect split: left = five 2.0s (Var = 0), right = five 8.0s (Var = 0)
     * VR = 10.0 - (0.5 * 0 + 0.5 * 0) = 10.0 */
    ht_var_stat_t lhs_perf, rhs_perf;
    ht_var_init(&lhs_perf);
    ht_var_init(&rhs_perf);
    for (int i = 0; i < 5; i++) {
        ht_var_update(&lhs_perf, (ht_float_t)2.0, (ht_float_t)1.0);
        ht_var_update(&rhs_perf, (ht_float_t)8.0, (ht_float_t)1.0);
    }
    ht_float_t vr_perf = ht_compute_variance_reduction(&pre_var, &lhs_perf, &rhs_perf,
                                                       (ht_float_t)5.0, (ht_float_t)0.01);
    HT_ASSERT_FLOAT_NEAR(vr_perf, (ht_float_t)10.0, 1e-5);

    /* Imperfect split:
     * left = four 2.0s + one 8.0 -> mean = 3.2, M2 = 28.8, Var = 7.2
     * right = one 2.0 + four 8.0s -> mean = 6.8, M2 = 28.8, Var = 7.2
     * VR = 10.0 - (0.5 * 7.2 + 0.5 * 7.2) = 10.0 - 7.2 = 2.8 */
    ht_var_stat_t lhs_imp, rhs_imp;
    ht_var_init(&lhs_imp);
    ht_var_init(&rhs_imp);
    for (int i = 0; i < 4; i++) {
        ht_var_update(&lhs_imp, (ht_float_t)2.0, (ht_float_t)1.0);
        ht_var_update(&rhs_imp, (ht_float_t)8.0, (ht_float_t)1.0);
    }
    ht_var_update(&lhs_imp, (ht_float_t)8.0, (ht_float_t)1.0);
    ht_var_update(&rhs_imp, (ht_float_t)2.0, (ht_float_t)1.0);
    ht_float_t vr_imp = ht_compute_variance_reduction(&pre_var, &lhs_imp, &rhs_imp,
                                                      (ht_float_t)5.0, (ht_float_t)0.01);
    HT_ASSERT_FLOAT_NEAR(vr_imp, (ht_float_t)2.8, 1e-5);

    /* Filter 1: min_samples_split rejection */
    /* lhs has 4 samples, min_samples_split = 5 -> rejected (-HT_INFINITY) */
    ht_var_stat_t lhs_small, rhs_large;
    ht_var_init(&lhs_small);
    ht_var_init(&rhs_large);
    for (int i = 0; i < 4; i++) {
        ht_var_update(&lhs_small, (ht_float_t)2.0, (ht_float_t)1.0);
    }
    for (int i = 0; i < 6; i++) {
        ht_var_update(&rhs_large, (ht_float_t)8.0, (ht_float_t)1.0);
    }
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_var, &lhs_small, &rhs_large,
                                              (ht_float_t)5.0, (ht_float_t)0.01),
                 -HT_INFINITY);

    /* Allowed when min_samples_split = 4 */
    HT_ASSERT(ht_compute_variance_reduction(&pre_var, &lhs_small, &rhs_large,
                                            (ht_float_t)4.0, (ht_float_t)0.01) > -HT_INFINITY);

    /* Filter 2: min_branch_fraction rejection */
    /* 100 samples total, lhs has 4 samples (4%) */
    ht_var_stat_t pre_100, lhs_4, rhs_96;
    ht_var_init(&pre_100);
    ht_var_init(&lhs_4);
    ht_var_init(&rhs_96);
    for (int i = 0; i < 4; i++) {
        ht_var_update(&pre_100, (ht_float_t)1.0, (ht_float_t)1.0);
        ht_var_update(&lhs_4, (ht_float_t)1.0, (ht_float_t)1.0);
    }
    for (int i = 0; i < 96; i++) {
        ht_var_update(&pre_100, (ht_float_t)10.0, (ht_float_t)1.0);
        ht_var_update(&rhs_96, (ht_float_t)10.0, (ht_float_t)1.0);
    }
    /* With min_branch_frac = 0.05, 4/100 = 0.04 < 0.05 -> rejected */
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_100, &lhs_4, &rhs_96,
                                              (ht_float_t)1.0, (ht_float_t)0.05),
                 -HT_INFINITY);
    /* With min_branch_frac = 0.03, 4/100 = 0.04 >= 0.03 -> accepted */
    HT_ASSERT(ht_compute_variance_reduction(&pre_100, &lhs_4, &rhs_96,
                                            (ht_float_t)1.0, (ht_float_t)0.03) > -HT_INFINITY);

    /* Filter 3: Degrees of freedom (pre_var->count <= 2.0) */
    ht_var_stat_t pre_small, l_one, r_one;
    ht_var_init(&pre_small);
    ht_var_init(&l_one);
    ht_var_init(&r_one);
    ht_var_update(&pre_small, (ht_float_t)1.0, (ht_float_t)1.0);
    ht_var_update(&pre_small, (ht_float_t)2.0, (ht_float_t)1.0);
    ht_var_update(&l_one, (ht_float_t)1.0, (ht_float_t)1.0);
    ht_var_update(&r_one, (ht_float_t)2.0, (ht_float_t)1.0);
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_small, &l_one, &r_one,
                                              (ht_float_t)1.0, (ht_float_t)0.01),
                 -HT_INFINITY);

    /* NULL arguments */
    HT_ASSERT_EQ(ht_compute_variance_reduction(NULL, &lhs_perf, &rhs_perf, (ht_float_t)1.0, (ht_float_t)0.01), -HT_INFINITY);
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_var, NULL, &rhs_perf, (ht_float_t)1.0, (ht_float_t)0.01), -HT_INFINITY);
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_var, &lhs_perf, NULL, (ht_float_t)1.0, (ht_float_t)0.01), -HT_INFINITY);
}

static void test_should_split_reg_analytical(void) {
    ht_split_candidate_t best, second_best;
    ht_split_candidate_init(&best);
    ht_split_candidate_init(&second_best);

    ht_float_t tau = (ht_float_t)0.05;

    /* 1. Invalid best candidate -> cannot split */
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, (ht_float_t)0.1, tau));

    /* Best has non-positive merit -> cannot split */
    best.merit = (ht_float_t)0.0;
    best.feature_idx = 0;
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, (ht_float_t)0.1, tau));
    best.merit = (ht_float_t)-1.5;
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, (ht_float_t)0.1, tau));

    /* 2. Best candidate valid (merit > 0), second_best invalid/null -> split! */
    best.merit = (ht_float_t)10.0;
    best.feature_idx = 0;
    HT_ASSERT(ht_should_split_reg(&best, NULL, (ht_float_t)0.5, tau));
    HT_ASSERT(ht_should_split_reg(&best, &second_best, (ht_float_t)0.5, tau));

    /* Second best has merit <= 0.0 -> split! */
    second_best.feature_idx = 1;
    second_best.merit = (ht_float_t)0.0;
    HT_ASSERT(ht_should_split_reg(&best, &second_best, (ht_float_t)0.5, tau));
    second_best.merit = (ht_float_t)-5.0;
    HT_ASSERT(ht_should_split_reg(&best, &second_best, (ht_float_t)0.5, tau));

    /* 3. Both valid: m1 = 10.0, m2 = 4.0 -> ratio = 0.4.
     * With epsilon = 0.3 -> 1 - epsilon = 0.7 > 0.4 -> split! */
    second_best.merit = (ht_float_t)4.0;
    HT_ASSERT(ht_should_split_reg(&best, &second_best, (ht_float_t)0.3, tau));

    /* With epsilon = 0.7 -> 1 - epsilon = 0.3 < 0.4 and epsilon (0.7) >= tau (0.05) -> do NOT split */
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, (ht_float_t)0.7, tau));

    /* 4. Tie-breaking: m1 = 10.0, m2 = 9.8 -> ratio = 0.98.
     * epsilon = 0.04 < tau (0.05) -> tie-break split triggered! */
    second_best.merit = (ht_float_t)9.8;
    HT_ASSERT(ht_should_split_reg(&best, &second_best, (ht_float_t)0.04, tau));

    /* epsilon = 0.06 >= tau (0.05) and 0.98 >= 1 - 0.06 (0.94) -> do NOT split */
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, (ht_float_t)0.06, tau));

    /* 5. NULL pointers */
    HT_ASSERT(!ht_should_split_reg(NULL, NULL, (ht_float_t)0.1, tau));
}

static void test_evaluate_reg_numeric_and_categorical_observers(void) {
    /* 1. Numeric observer with sharp step function:
     * x in [0, 4] -> y = 2.0 (15 samples)
     * x in [6, 10] -> y = 20.0 (15 samples)
     * pre_var: 30 samples */
    ht_var_stat_t pre_var;
    ht_var_init(&pre_var);
    ht_reg_numeric_obs_t num_obs;
    ht_reg_numeric_init(&num_obs);

    for (int i = 0; i < 15; i++) {
        ht_float_t x_val = (ht_float_t)(i % 5); /* 0, 1, 2, 3, 4 */
        ht_float_t y_val = (ht_float_t)2.0;
        ht_var_update(&pre_var, y_val, (ht_float_t)1.0);
        ht_reg_numeric_update(&num_obs, x_val, y_val, (ht_float_t)1.0);
    }
    for (int i = 0; i < 15; i++) {
        ht_float_t x_val = (ht_float_t)(6 + (i % 5)); /* 6, 7, 8, 9, 10 */
        ht_float_t y_val = (ht_float_t)20.0;
        ht_var_update(&pre_var, y_val, (ht_float_t)1.0);
        ht_reg_numeric_update(&num_obs, x_val, y_val, (ht_float_t)1.0);
    }

    ht_split_candidate_t num_cand;
    ht_evaluate_reg_numeric_observer(&num_obs, &pre_var, 0, (ht_float_t)5.0, (ht_float_t)0.01, &num_cand);

    HT_ASSERT(num_cand.merit > (ht_float_t)50.0);
    HT_ASSERT_EQ(num_cand.feature_idx, 0);
    HT_ASSERT_EQ(num_cand.is_categorical, 0);
    /* Threshold must lie between lower slots and upper slots (~5.0) */
    HT_ASSERT(num_cand.split_val.threshold >= (ht_float_t)2.0 &&
              num_cand.split_val.threshold <= (ht_float_t)7.0);
    HT_ASSERT_FLOAT_NEAR(num_cand.lhs_var.count, (ht_float_t)15.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(num_cand.rhs_var.count, (ht_float_t)15.0, 1e-4);

    /* 2. Categorical observer:
     * cat 0 -> y = 0.0 (10 samples)
     * cat 1 -> y = 20.0 (10 samples)
     * cat 2 -> y = 20.0 (10 samples) */
    ht_var_stat_t cat_pre_var;
    ht_var_init(&cat_pre_var);
    ht_reg_categorical_obs_t cat_obs;
    ht_reg_categorical_init(&cat_obs);

    for (int i = 0; i < 10; i++) {
        ht_var_update(&cat_pre_var, (ht_float_t)0.0, (ht_float_t)1.0);
        ht_reg_categorical_update(&cat_obs, 0, (ht_float_t)0.0, (ht_float_t)1.0);
    }
    for (int i = 0; i < 10; i++) {
        ht_var_update(&cat_pre_var, (ht_float_t)20.0, (ht_float_t)1.0);
        ht_reg_categorical_update(&cat_obs, 1, (ht_float_t)20.0, (ht_float_t)1.0);
    }
    for (int i = 0; i < 10; i++) {
        ht_var_update(&cat_pre_var, (ht_float_t)20.0, (ht_float_t)1.0);
        ht_reg_categorical_update(&cat_obs, 2, (ht_float_t)20.0, (ht_float_t)1.0);
    }

    ht_split_candidate_t cat_cand;
    ht_evaluate_reg_categorical_observer(&cat_obs, &cat_pre_var, 1, (ht_float_t)5.0, (ht_float_t)0.01, &cat_cand);

    HT_ASSERT(cat_cand.merit > (ht_float_t)50.0);
    HT_ASSERT_EQ(cat_cand.feature_idx, 1);
    HT_ASSERT_EQ(cat_cand.is_categorical, 1);
    /* Best split must isolate cat 0 (y=0) from cats 1 and 2 (y=20) */
    HT_ASSERT_EQ(cat_cand.split_val.cat_val, 0);
    HT_ASSERT_FLOAT_NEAR(cat_cand.lhs_var.count, (ht_float_t)10.0, 1e-4);
    HT_ASSERT_FLOAT_NEAR(cat_cand.rhs_var.count, (ht_float_t)20.0, 1e-4);
}

static void test_find_best_reg_split_candidates_and_tie_breaking(void) {
    static ht_memory_pool_t pool;
    ht_memory_init(&pool);

    uint16_t stats_idx = ht_stats_alloc(&pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);
    ht_leaf_stats_t *stats = ht_stats_get(&pool, stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    ht_feature_type_t feat_types[HT_MAX_FEATURES];
    feat_types[0] = HT_FEAT_NUMERIC;      /* Feature 0: Strong split */
    feat_types[1] = HT_FEAT_NUMERIC;      /* Feature 1: Weak split */
    feat_types[2] = HT_FEAT_CATEGORICAL;  /* Feature 2: Moderate split */
    feat_types[3] = HT_FEAT_NUMERIC;      /* Feature 3: Constant feature (uninformative) */

    ht_var_stat_t pre_var;
    ht_var_init(&pre_var);

    for (uint8_t f = 0; f < 4; f++) {
        if (feat_types[f] == HT_FEAT_NUMERIC) {
            ht_reg_numeric_init(&stats->observers[f].reg_num);
        } else {
            ht_reg_categorical_init(&stats->observers[f].reg_cat);
        }
    }

    /* 40 samples */
    for (int i = 0; i < 40; i++) {
        ht_float_t x0 = (i < 20) ? (ht_float_t)2.0 : (ht_float_t)8.0;
        ht_float_t x1 = (i < 8)  ? (ht_float_t)1.0 : (ht_float_t)9.0;
        int32_t   x2 = (i < 15) ? 0 : 1;
        ht_float_t x3 = (ht_float_t)5.0; /* Constant */

        /* Target y strongly depends on x0, weakly on x2, minimally on x1 */
        ht_float_t y = (i < 20) ? (ht_float_t)10.0 : (ht_float_t)50.0;

        ht_var_update(&pre_var, y, (ht_float_t)1.0);
        ht_reg_numeric_update(&stats->observers[0].reg_num, x0, y, (ht_float_t)1.0);
        ht_reg_numeric_update(&stats->observers[1].reg_num, x1, y, (ht_float_t)1.0);
        ht_reg_categorical_update(&stats->observers[2].reg_cat, x2, y, (ht_float_t)1.0);
        ht_reg_numeric_update(&stats->observers[3].reg_num, x3, y, (ht_float_t)1.0);
    }

    ht_split_candidate_t best, second_best;
    ht_find_best_reg_split_candidates(stats, &pre_var, feat_types, 4,
                                      (ht_float_t)5.0, (ht_float_t)0.01, true,
                                      &best, &second_best);

    /* Best must be Feature 0 (strongest variance reduction) */
    HT_ASSERT_EQ(best.feature_idx, 0);
    HT_ASSERT(best.merit > (ht_float_t)100.0);

    /* Second best must be Feature 1 or Feature 2 with valid positive merit */
    HT_ASSERT(second_best.feature_idx != HT_NO_FEATURE);
    HT_ASSERT(second_best.merit > (ht_float_t)0.0);
    HT_ASSERT(second_best.merit <= best.merit);

    /* Tie-breaking parity test: Feature 0 and Feature 1 have identical merit */
    ht_leaf_stats_t *tie_stats = ht_stats_get(&pool, stats_idx);
    ht_reg_numeric_init(&tie_stats->observers[0].reg_num);
    ht_reg_numeric_init(&tie_stats->observers[1].reg_num);

    for (int i = 0; i < 20; i++) {
        ht_reg_numeric_update(&tie_stats->observers[0].reg_num, (ht_float_t)1.0, (ht_float_t)5.0, (ht_float_t)1.0);
        ht_reg_numeric_update(&tie_stats->observers[1].reg_num, (ht_float_t)1.0, (ht_float_t)5.0, (ht_float_t)1.0);
    }
    for (int i = 0; i < 20; i++) {
        ht_reg_numeric_update(&tie_stats->observers[0].reg_num, (ht_float_t)9.0, (ht_float_t)45.0, (ht_float_t)1.0);
        ht_reg_numeric_update(&tie_stats->observers[1].reg_num, (ht_float_t)9.0, (ht_float_t)45.0, (ht_float_t)1.0);
    }

    ht_feature_type_t tie_types[HT_MAX_FEATURES];
    tie_types[0] = HT_FEAT_NUMERIC;
    tie_types[1] = HT_FEAT_NUMERIC;
    ht_find_best_reg_split_candidates(tie_stats, &pre_var, tie_types, 2,
                                      (ht_float_t)5.0, (ht_float_t)0.01, true,
                                      &best, &second_best);

    /* On exact merit tie, higher feature index (Feature 1) wins matching River Timsort */
    HT_ASSERT_FLOAT_NEAR(best.merit, second_best.merit, 1e-5);
    HT_ASSERT_EQ(best.feature_idx, 1);
    HT_ASSERT_EQ(second_best.feature_idx, 0);

    ht_stats_free(&pool, stats_idx);
}

static void test_reg_criterion_vtable_and_null_edge_cases(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_VARIANCE_REDUCTION);
    HT_ASSERT_NOT_NULL(crit);
    ht_float_t dummy_dist[HT_MAX_CLASSES] = { (ht_float_t)10.0, (ht_float_t)10.0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dummy_dist, 2), (ht_float_t)1.0, 1e-6);
    HT_ASSERT_EQ(crit->merit_of_split(dummy_dist, dummy_dist, dummy_dist, 2, (ht_float_t)0.01), -HT_INFINITY);

    /* Null guards */
    ht_split_candidate_t cand;
    ht_evaluate_reg_numeric_observer(NULL, NULL, 0, (ht_float_t)1.0, (ht_float_t)0.01, &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_evaluate_reg_categorical_observer(NULL, NULL, 0, (ht_float_t)1.0, (ht_float_t)0.01, &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_split_candidate_t best, second;
    ht_find_best_reg_split_candidates(NULL, NULL, NULL, 0, (ht_float_t)1.0, (ht_float_t)0.01, true, &best, &second);
    HT_ASSERT_EQ(best.merit, -HT_INFINITY);
    HT_ASSERT_EQ(second.merit, -HT_INFINITY);
}

/* ========================================================================= */
/* --- Test Runner Entry Point --------------------------------------------- */
/* ========================================================================= */

void test_criterion_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Criterios de Impureza & Hoeffding Bound (ht_criterion)");
    HT_RUN_TEST(test_gini_impurity_analytical);
    HT_RUN_TEST(test_gini_merit_and_min_branch);
    HT_RUN_TEST(test_entropy_analytical);
    HT_RUN_TEST(test_info_gain_merit_and_range);
    HT_RUN_TEST(test_hoeffding_bound_analytical);
    HT_RUN_TEST(test_evaluate_numeric_observer);
    HT_RUN_TEST(test_evaluate_categorical_observer);
    HT_RUN_TEST(test_find_best_split_candidates_and_tie_breaking);
    HT_RUN_TEST(test_should_split_logic);
    HT_RUN_TEST(test_criterion_edge_cases_and_null);
    HT_RUN_TEST(test_vr_analytical_and_filters);
    HT_RUN_TEST(test_should_split_reg_analytical);
    HT_RUN_TEST(test_evaluate_reg_numeric_and_categorical_observers);
    HT_RUN_TEST(test_find_best_reg_split_candidates_and_tie_breaking);
    HT_RUN_TEST(test_reg_criterion_vtable_and_null_edge_cases);
}
