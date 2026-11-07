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
    ht_gaussian_obs_t obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; c++) {
        ht_gaussian_init(&obs[c]);
    }

    /* Populate Class 0: values in [0.0, 4.0] (mean 2.0) */
    for (int i = 0; i <= 20; i++) {
        ht_gaussian_update(obs, 0, (ht_float_t)(i * 0.2), (ht_float_t)1.0);
    }
    /* Populate Class 1: values in [6.0, 10.0] (mean 8.0) */
    for (int i = 0; i <= 20; i++) {
        ht_gaussian_update(obs, 1, (ht_float_t)(6.0 + i * 0.2), (ht_float_t)1.0);
    }

    ht_float_t pre_dist[HT_MAX_CLASSES] = { (ht_float_t)21.0, (ht_float_t)21.0, 0, 0 };
    ht_split_candidate_t cand;

    ht_evaluate_numeric_observer(obs, pre_dist, 2, 0, crit, (ht_float_t)0.01, &cand);

    HT_ASSERT_EQ(cand.feature_idx, 0);
    HT_ASSERT_EQ(cand.is_categorical, 0);
    HT_ASSERT(cand.merit > (ht_float_t)0.8); /* High merit because well-separated */
    /* Threshold should be in between [4.0, 6.0] */
    HT_ASSERT(cand.split_val.threshold >= (ht_float_t)3.5 && cand.split_val.threshold <= (ht_float_t)6.5);

    /* Test constant feature: all samples equal to 5.0 */
    ht_gaussian_obs_t const_obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; c++) {
        ht_gaussian_init(&const_obs[c]);
    }
    for (int i = 0; i < 20; i++) {
        ht_gaussian_update(const_obs, 0, (ht_float_t)5.0, (ht_float_t)1.0);
        ht_gaussian_update(const_obs, 1, (ht_float_t)5.0, (ht_float_t)1.0);
    }
    ht_split_candidate_t const_cand;
    ht_evaluate_numeric_observer(const_obs, pre_dist, 2, 1, crit, (ht_float_t)0.01, &const_cand);
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
        ht_gaussian_update(stats->observers[0].num, 0, (ht_float_t)(4.0 + i * 0.04), (ht_float_t)1.0);
        ht_gaussian_update(stats->observers[0].num, 1, (ht_float_t)(4.0 + i * 0.04), (ht_float_t)1.0);
    }

    /* Populate Feature 1: class 0 in [0..2], class 1 in [8..10] */
    for (int i = 0; i < 50; i++) {
        ht_gaussian_update(stats->observers[1].num, 0, (ht_float_t)(i * 0.04), (ht_float_t)1.0);
        ht_gaussian_update(stats->observers[1].num, 1, (ht_float_t)(8.0 + i * 0.04), (ht_float_t)1.0);
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
}
