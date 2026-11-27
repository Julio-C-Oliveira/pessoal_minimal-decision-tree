/**
 * @file test_criterion.c
 * @brief Level 1 & Level 2 tests for split criteria (Gini, InfoGain, Variance Reduction),
 *        Hoeffding bound, and candidate search across float32, double64, and Q16.16.
 */

#include "test_harness.h"
#include "ht_criterion.h"
#include "ht_memory.h"
#include "ht_observer.h"

/* ========================================================================= */
/* --- Level 1: Analytical Gini Impurity Tests ----------------------------- */
/* ========================================================================= */

static void test_gini_impurity_analytical(void) {
    ht_float_t pure_dist[HT_MAX_CLASSES] = { HT_CONST(100.0), 0, 0, 0 };
    ht_float_t g_pure = ht_compute_gini(pure_dist, 2, HT_CONST(100.0));
    HT_ASSERT_FLOAT_NEAR(g_pure, HT_CONST(0.0), HT_TEST_TOL(1e-6));

    ht_float_t eq2_dist[HT_MAX_CLASSES] = { HT_CONST(50.0), HT_CONST(50.0), 0, 0 };
    ht_float_t g_eq2 = ht_compute_gini(eq2_dist, 2, HT_CONST(100.0));
    HT_ASSERT_FLOAT_NEAR(g_eq2, HT_CONST(0.5), HT_TEST_TOL(1e-6));

    ht_float_t eq4_dist[HT_MAX_CLASSES] = {
        HT_CONST(25.0), HT_CONST(25.0), HT_CONST(25.0), HT_CONST(25.0)
    };
    ht_float_t g_eq4 = ht_compute_gini(eq4_dist, 4, HT_CONST(100.0));
    HT_ASSERT_FLOAT_NEAR(g_eq4, HT_CONST(0.75), HT_TEST_TOL(1e-6));

    ht_float_t skew_dist[HT_MAX_CLASSES] = { HT_CONST(90.0), HT_CONST(10.0), 0, 0 };
    ht_float_t g_skew = ht_compute_gini(skew_dist, 2, HT_CONST(100.0));
    HT_ASSERT_FLOAT_NEAR(g_skew, HT_CONST(0.18), HT_TEST_TOL(1e-6));

    ht_float_t empty_dist[HT_MAX_CLASSES] = { 0, 0, 0, 0 };
    ht_float_t g_empty = ht_compute_gini(empty_dist, 2, HT_CONST(0.0));
    HT_ASSERT_FLOAT_NEAR(g_empty, HT_CONST(1.0), HT_TEST_TOL(1e-6));
}

static void test_gini_merit_and_min_branch(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);
    HT_ASSERT_NOT_NULL(crit);

    ht_float_t pre_dist[HT_MAX_CLASSES] = { HT_CONST(50.0), HT_CONST(50.0), 0, 0 };

    ht_float_t lhs_perf[HT_MAX_CLASSES] = { HT_CONST(50.0), 0, 0, 0 };
    ht_float_t rhs_perf[HT_MAX_CLASSES] = { 0, HT_CONST(50.0), 0, 0 };
    ht_float_t m_perf = crit->merit_of_split(pre_dist, lhs_perf, rhs_perf, 2, HT_CONST(0.01));
    HT_ASSERT_FLOAT_NEAR(m_perf, HT_CONST(1.0), HT_TEST_TOL(1e-6));

    ht_float_t lhs_use[HT_MAX_CLASSES] = { HT_CONST(25.0), HT_CONST(25.0), 0, 0 };
    ht_float_t rhs_use[HT_MAX_CLASSES] = { HT_CONST(25.0), HT_CONST(25.0), 0, 0 };
    ht_float_t m_use = crit->merit_of_split(pre_dist, lhs_use, rhs_use, 2, HT_CONST(0.01));
    HT_ASSERT_FLOAT_NEAR(m_use, HT_CONST(0.5), HT_TEST_TOL(1e-6));

    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(pre_dist, 2), HT_CONST(1.0), HT_TEST_TOL(1e-6));

    ht_float_t lhs_small[HT_MAX_CLASSES] = { HT_CONST(3.0), 0, 0, 0 };
    ht_float_t rhs_large[HT_MAX_CLASSES] = { HT_CONST(47.0), HT_CONST(50.0), 0, 0 };
    ht_float_t m_rej = crit->merit_of_split(pre_dist, lhs_small, rhs_large, 2, HT_CONST(0.05));
    HT_ASSERT_EQ(m_rej, -HT_INFINITY);

    ht_float_t lhs_exact[HT_MAX_CLASSES] = { HT_CONST(5.0), 0, 0, 0 };
    ht_float_t rhs_exact[HT_MAX_CLASSES] = { HT_CONST(45.0), HT_CONST(50.0), 0, 0 };
    ht_float_t m_bnd = crit->merit_of_split(pre_dist, lhs_exact, rhs_exact, 2, HT_CONST(0.05));
    HT_ASSERT_EQ(m_bnd, -HT_INFINITY);

    ht_float_t lhs_valid[HT_MAX_CLASSES] = { HT_CONST(6.0), 0, 0, 0 };
    ht_float_t rhs_valid[HT_MAX_CLASSES] = { HT_CONST(44.0), HT_CONST(50.0), 0, 0 };
    ht_float_t m_val = crit->merit_of_split(pre_dist, lhs_valid, rhs_valid, 2, HT_CONST(0.05));
    HT_ASSERT(m_val > -HT_INFINITY);

    ht_float_t zero_dist[HT_MAX_CLASSES] = { 0, 0, 0, 0 };
    HT_ASSERT_EQ(crit->merit_of_split(pre_dist, zero_dist, zero_dist, 2, HT_CONST(0.01)), -HT_INFINITY);
}

/* ========================================================================= */
/* --- Level 1: Analytical Information Gain Tests -------------------------- */
/* ========================================================================= */

static void test_entropy_analytical(void) {
    ht_float_t pure_dist[HT_MAX_CLASSES] = { HT_CONST(100.0), 0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(pure_dist, 2), HT_CONST(0.0), HT_TEST_TOL(1e-6));

    ht_float_t eq2_dist[HT_MAX_CLASSES] = { HT_CONST(50.0), HT_CONST(50.0), 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(eq2_dist, 2), HT_CONST(1.0), HT_TEST_TOL(1e-6));

    ht_float_t eq4_dist[HT_MAX_CLASSES] = {
        HT_CONST(25.0), HT_CONST(25.0), HT_CONST(25.0), HT_CONST(25.0)
    };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(eq4_dist, 4), HT_CONST(2.0), HT_TEST_TOL(1e-6));

    ht_float_t skew_dist[HT_MAX_CLASSES] = { HT_CONST(80.0), HT_CONST(20.0), 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(skew_dist, 2), HT_CONST(0.72192809), HT_TEST_TOL(1e-5));

    ht_float_t empty_dist[HT_MAX_CLASSES] = { 0, 0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(ht_compute_entropy(empty_dist, 2), HT_CONST(0.0), HT_TEST_TOL(1e-6));
}

static void test_info_gain_merit_and_range(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_INFO_GAIN);
    HT_ASSERT_NOT_NULL(crit);

    ht_float_t pre_dist[HT_MAX_CLASSES] = { HT_CONST(50.0), HT_CONST(50.0), 0, 0 };

    ht_float_t lhs_perf[HT_MAX_CLASSES] = { HT_CONST(50.0), 0, 0, 0 };
    ht_float_t rhs_perf[HT_MAX_CLASSES] = { 0, HT_CONST(50.0), 0, 0 };
    ht_float_t m_perf = crit->merit_of_split(pre_dist, lhs_perf, rhs_perf, 2, HT_CONST(0.01));
    HT_ASSERT_FLOAT_NEAR(m_perf, HT_CONST(1.0), HT_TEST_TOL(1e-6));

    ht_float_t lhs_use[HT_MAX_CLASSES] = { HT_CONST(25.0), HT_CONST(25.0), 0, 0 };
    ht_float_t rhs_use[HT_MAX_CLASSES] = { HT_CONST(25.0), HT_CONST(25.0), 0, 0 };
    ht_float_t m_use = crit->merit_of_split(pre_dist, lhs_use, rhs_use, 2, HT_CONST(0.01));
    HT_ASSERT_FLOAT_NEAR(m_use, HT_CONST(0.0), HT_TEST_TOL(1e-6));

    ht_float_t dist_1c[HT_MAX_CLASSES] = { HT_CONST(100.0), 0, 0, 0 };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dist_1c, 4), HT_CONST(1.0), HT_TEST_TOL(1e-6));

    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(pre_dist, 4), HT_CONST(1.0), HT_TEST_TOL(1e-6));

    ht_float_t dist_3c[HT_MAX_CLASSES] = { HT_CONST(30.0), HT_CONST(30.0), HT_CONST(40.0), 0 };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dist_3c, 4), HT_CONST(1.5849625), HT_TEST_TOL(1e-5));

    ht_float_t dist_4c[HT_MAX_CLASSES] = {
        HT_CONST(25.0), HT_CONST(25.0), HT_CONST(25.0), HT_CONST(25.0)
    };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dist_4c, 4), HT_CONST(2.0), HT_TEST_TOL(1e-6));

    ht_float_t lhs_small[HT_MAX_CLASSES] = { HT_CONST(2.0), 0, 0, 0 };
    ht_float_t rhs_large[HT_MAX_CLASSES] = { HT_CONST(48.0), HT_CONST(50.0), 0, 0 };
    HT_ASSERT_EQ(crit->merit_of_split(pre_dist, lhs_small, rhs_large, 2, HT_CONST(0.05)), -HT_INFINITY);
}

/* ========================================================================= */
/* --- Level 1: Analytical Hoeffding Bound Tests --------------------------- */
/* ========================================================================= */

static void test_hoeffding_bound_analytical(void) {
    /* Exact delta = 1e-7 via ln(1/delta) = 7 * ln(10) = 16.11809565 across ALL 3 modes */
    ht_float_t neg_ln_1e7 = HT_MUL(HT_CONST(7.0), HT_LN10);
    ht_float_t eps_200 = ht_hoeffding_bound_ln(HT_CONST(1.0), neg_ln_1e7, HT_CONST(200.0));
    HT_ASSERT_FLOAT_NEAR(eps_200, HT_CONST(0.20073674), HT_TEST_TOL(1e-4));

    ht_float_t eps_r2 = ht_hoeffding_bound_ln(HT_CONST(2.0), neg_ln_1e7, HT_CONST(200.0));
    HT_ASSERT_FLOAT_NEAR(eps_r2, HT_CONST(0.40147348), HT_TEST_TOL(1e-4));

    ht_float_t eps_2000 = ht_hoeffding_bound_ln(HT_CONST(1.0), neg_ln_1e7, HT_CONST(2000.0));
    HT_ASSERT(eps_2000 < eps_200);
    HT_ASSERT_FLOAT_NEAR(eps_2000, HT_CONST(0.06347853), HT_TEST_TOL(1e-4));

    /* Direct ht_hoeffding_bound with representable delta = 0.001 (ln(1000) = 6.907755)
     * For R = 1, n = 200: sqrt(6.907755 / 400) = 0.1314130 */
    ht_float_t eps_1e3 = ht_hoeffding_bound(HT_CONST(1.0), HT_CONST(0.001), HT_CONST(200.0));
    HT_ASSERT_FLOAT_NEAR(eps_1e3, HT_CONST(0.1314130), HT_TEST_TOL(2e-3));

#ifndef HT_USE_FIXED_POINT
    ht_float_t eps_direct_1e7 = ht_hoeffding_bound(HT_CONST(1.0), HT_CONST(1e-7), HT_CONST(200.0));
    HT_ASSERT_FLOAT_NEAR(eps_direct_1e7, HT_CONST(0.20073674), HT_TEST_TOL(1e-4));
#endif

    /* Edge cases */
    HT_ASSERT_EQ(ht_hoeffding_bound(HT_CONST(1.0), HT_CONST(0.001), HT_CONST(0.0)), HT_INFINITY);
    HT_ASSERT_EQ(ht_hoeffding_bound(HT_CONST(1.0), HT_CONST(0.001), HT_CONST(-5.0)), HT_INFINITY);
    HT_ASSERT_EQ(ht_hoeffding_bound(HT_CONST(1.0), HT_CONST(0.0), HT_CONST(100.0)), HT_INFINITY);
    HT_ASSERT_FLOAT_EQ(ht_hoeffding_bound(HT_CONST(1.0), HT_CONST(1.0), HT_CONST(100.0)), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_hoeffding_bound(HT_CONST(1.0), HT_CONST(1.5), HT_CONST(100.0)), HT_CONST(0.0));
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

    for (int i = 0; i <= 20; i++) {
        ht_gaussian_update(obs, 0, HT_MUL(HT_FROM_INT(i), HT_CONST(0.2)), HT_CONST(1.0));
    }
    for (int i = 0; i <= 20; i++) {
        ht_gaussian_update(obs, 1, HT_ADD(HT_CONST(6.0), HT_MUL(HT_FROM_INT(i), HT_CONST(0.2))), HT_CONST(1.0));
    }

    ht_float_t pre_dist[HT_MAX_CLASSES] = { HT_CONST(21.0), HT_CONST(21.0), 0, 0 };
    ht_split_candidate_t cand;

    ht_evaluate_numeric_observer(obs, pre_dist, 2, 0, crit, HT_CONST(0.01), &cand);

    HT_ASSERT_EQ(cand.feature_idx, 0);
    HT_ASSERT_EQ(cand.is_categorical, 0);
    HT_ASSERT(cand.merit > HT_CONST(0.8));
    HT_ASSERT(cand.split_val.threshold >= HT_CONST(3.5) && cand.split_val.threshold <= HT_CONST(6.5));

    ht_gaussian_obs_t const_obs[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; c++) {
        ht_gaussian_init(&const_obs[c]);
    }
    for (int i = 0; i < 20; i++) {
        ht_gaussian_update(const_obs, 0, HT_CONST(5.0), HT_CONST(1.0));
        ht_gaussian_update(const_obs, 1, HT_CONST(5.0), HT_CONST(1.0));
    }
    ht_split_candidate_t const_cand;
    ht_evaluate_numeric_observer(const_obs, pre_dist, 2, 1, crit, HT_CONST(0.01), &const_cand);
    HT_ASSERT_EQ(const_cand.merit, -HT_INFINITY);
    HT_ASSERT_EQ(const_cand.feature_idx, HT_NO_FEATURE);
}

static void test_evaluate_categorical_observer(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);
    ht_categorical_obs_t cat_obs;
    ht_categorical_init(&cat_obs);

    ht_categorical_update(&cat_obs, 0, 1, HT_CONST(50.0));
    ht_categorical_update(&cat_obs, 1, 3, HT_CONST(50.0));

    ht_float_t pre_dist[HT_MAX_CLASSES] = { HT_CONST(50.0), HT_CONST(50.0), 0, 0 };
    ht_split_candidate_t cand;

    ht_evaluate_categorical_observer(&cat_obs, pre_dist, 2, 2, crit, HT_CONST(0.01), &cand);

    HT_ASSERT_EQ(cand.feature_idx, 2);
    HT_ASSERT_EQ(cand.is_categorical, 1);
    HT_ASSERT_FLOAT_NEAR(cand.merit, HT_CONST(1.0), HT_TEST_TOL(1e-6));
    HT_ASSERT(cand.split_val.cat_val == 1 || cand.split_val.cat_val == 3);

    ht_categorical_obs_t empty_cat;
    ht_categorical_init(&empty_cat);
    ht_split_candidate_t empty_cand;
    ht_evaluate_categorical_observer(&empty_cat, pre_dist, 2, 3, crit, HT_CONST(0.01), &empty_cand);
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
    ht_leaf_stats_t *stats = ht_stats_get(&pool, stats_idx);

    ht_feature_type_t types[HT_MAX_FEATURES];
    types[0] = HT_FEAT_NUMERIC;
    types[1] = HT_FEAT_NUMERIC;
    types[2] = HT_FEAT_CATEGORICAL;

    for (int i = 0; i < 50; i++) {
        ht_float_t v = HT_ADD(HT_CONST(4.0), HT_MUL(HT_FROM_INT(i), HT_CONST(0.04)));
        ht_gaussian_update(stats->observers[0].num, 0, v, HT_CONST(1.0));
        ht_gaussian_update(stats->observers[0].num, 1, v, HT_CONST(1.0));
    }

    for (int i = 0; i < 50; i++) {
        ht_gaussian_update(stats->observers[1].num, 0, HT_MUL(HT_FROM_INT(i), HT_CONST(0.04)), HT_CONST(1.0));
        ht_gaussian_update(stats->observers[1].num, 1, HT_ADD(HT_CONST(8.0), HT_MUL(HT_FROM_INT(i), HT_CONST(0.04))), HT_CONST(1.0));
    }

    ht_categorical_update(&stats->observers[2].cat, 0, 0, HT_CONST(50.0));
    ht_categorical_update(&stats->observers[2].cat, 1, 1, HT_CONST(50.0));

    ht_float_t pre_dist[HT_MAX_CLASSES] = { HT_CONST(50.0), HT_CONST(50.0), 0, 0 };
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_GINI);

    ht_split_candidate_t best, second_best;
    ht_find_best_split_candidates(stats, pre_dist, types, 3, 2, crit,
                                  HT_CONST(0.01), true, &best, &second_best);

    HT_ASSERT_FLOAT_NEAR(best.merit, HT_CONST(1.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(second_best.merit, HT_CONST(1.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_EQ(best.feature_idx, 2);
    HT_ASSERT_EQ(second_best.feature_idx, 1);

    ht_stats_free(&pool, stats_idx);
}

static void test_should_split_logic(void) {
    ht_split_candidate_t best, second_best;
    ht_split_candidate_init(&best);
    ht_split_candidate_init(&second_best);

    ht_float_t tau = HT_CONST(0.05);

    HT_ASSERT(!ht_should_split(&best, &second_best, HT_CONST(0.1), tau));

    best.merit = HT_CONST(0.8);
    best.feature_idx = 0;
    HT_ASSERT(ht_should_split(&best, &second_best, HT_CONST(0.5), tau));

    second_best.merit = HT_CONST(0.5);
    second_best.feature_idx = 1;
    HT_ASSERT(ht_should_split(&best, &second_best, HT_CONST(0.2), tau));

    HT_ASSERT(!ht_should_split(&best, &second_best, HT_CONST(0.4), tau));

    best.merit = HT_CONST(0.81);
    second_best.merit = HT_CONST(0.80);
    HT_ASSERT(ht_should_split(&best, &second_best, HT_CONST(0.04), tau));

    best.merit = -HT_INFINITY;
    best.feature_idx = HT_NO_FEATURE;
    second_best.merit = -HT_INFINITY;
    second_best.feature_idx = HT_NO_FEATURE;
    HT_ASSERT(!ht_should_split(&best, &second_best, HT_CONST(0.01), tau));
}

static void test_criterion_edge_cases_and_null(void) {
    HT_ASSERT_NULL(ht_criterion_get_vtable((ht_criterion_type_t)999));

    ht_split_candidate_t cand;
    ht_evaluate_numeric_observer(NULL, NULL, 0, 0, NULL, HT_CONST(0.01), &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_evaluate_categorical_observer(NULL, NULL, 0, 0, NULL, HT_CONST(0.01), &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_split_candidate_t best, second;
    ht_find_best_split_candidates(NULL, NULL, NULL, 0, 0, NULL, HT_CONST(0.01), true, &best, &second);
    HT_ASSERT_EQ(best.merit, -HT_INFINITY);
    HT_ASSERT_EQ(second.merit, -HT_INFINITY);

    HT_ASSERT(!ht_should_split(NULL, NULL, HT_CONST(0.1), HT_CONST(0.05)));
}

/* ========================================================================= */
/* --- Level 1 & 2: Regression Variance Reduction Split Tests -------------- */
/* ========================================================================= */

static void test_vr_analytical_and_filters(void) {
    ht_var_stat_t pre_var;
    ht_var_init(&pre_var);
    for (int i = 0; i < 5; i++) {
        ht_var_update(&pre_var, HT_CONST(2.0), HT_CONST(1.0));
        ht_var_update(&pre_var, HT_CONST(8.0), HT_CONST(1.0));
    }
    HT_ASSERT_FLOAT_NEAR(pre_var.count, HT_CONST(10.0), HT_TEST_TOL(1e-6));
    HT_ASSERT_FLOAT_NEAR(pre_var.mean, HT_CONST(5.0), HT_TEST_TOL(1e-6));
    HT_ASSERT_FLOAT_NEAR(ht_var_value(&pre_var), HT_CONST(10.0), HT_TEST_TOL(1e-2));

    ht_var_stat_t lhs_perf, rhs_perf;
    ht_var_init(&lhs_perf);
    ht_var_init(&rhs_perf);
    for (int i = 0; i < 5; i++) {
        ht_var_update(&lhs_perf, HT_CONST(2.0), HT_CONST(1.0));
        ht_var_update(&rhs_perf, HT_CONST(8.0), HT_CONST(1.0));
    }
    ht_float_t vr_perf = ht_compute_variance_reduction(&pre_var, &lhs_perf, &rhs_perf,
                                                       HT_CONST(5.0), HT_CONST(0.01));
    HT_ASSERT_FLOAT_NEAR(vr_perf, HT_CONST(10.0), HT_TEST_TOL(1e-2));

    ht_var_stat_t lhs_imp, rhs_imp;
    ht_var_init(&lhs_imp);
    ht_var_init(&rhs_imp);
    for (int i = 0; i < 4; i++) {
        ht_var_update(&lhs_imp, HT_CONST(2.0), HT_CONST(1.0));
        ht_var_update(&rhs_imp, HT_CONST(8.0), HT_CONST(1.0));
    }
    ht_var_update(&lhs_imp, HT_CONST(8.0), HT_CONST(1.0));
    ht_var_update(&rhs_imp, HT_CONST(2.0), HT_CONST(1.0));
    ht_float_t vr_imp = ht_compute_variance_reduction(&pre_var, &lhs_imp, &rhs_imp,
                                                      HT_CONST(5.0), HT_CONST(0.01));
    HT_ASSERT_FLOAT_NEAR(vr_imp, HT_CONST(2.8), HT_TEST_TOL(2e-2));

    ht_var_stat_t lhs_small, rhs_large;
    ht_var_init(&lhs_small);
    ht_var_init(&rhs_large);
    for (int i = 0; i < 4; i++) {
        ht_var_update(&lhs_small, HT_CONST(2.0), HT_CONST(1.0));
    }
    for (int i = 0; i < 6; i++) {
        ht_var_update(&rhs_large, HT_CONST(8.0), HT_CONST(1.0));
    }
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_var, &lhs_small, &rhs_large,
                                               HT_CONST(5.0), HT_CONST(0.01)),
                 -HT_INFINITY);

    HT_ASSERT(ht_compute_variance_reduction(&pre_var, &lhs_small, &rhs_large,
                                            HT_CONST(4.0), HT_CONST(0.01)) > -HT_INFINITY);

    ht_var_stat_t pre_100, lhs_4, rhs_96;
    ht_var_init(&pre_100);
    ht_var_init(&lhs_4);
    ht_var_init(&rhs_96);
    for (int i = 0; i < 4; i++) {
        ht_var_update(&pre_100, HT_CONST(1.0), HT_CONST(1.0));
        ht_var_update(&lhs_4, HT_CONST(1.0), HT_CONST(1.0));
    }
    for (int i = 0; i < 96; i++) {
        ht_var_update(&pre_100, HT_CONST(10.0), HT_CONST(1.0));
        ht_var_update(&rhs_96, HT_CONST(10.0), HT_CONST(1.0));
    }
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_100, &lhs_4, &rhs_96,
                                               HT_CONST(1.0), HT_CONST(0.05)),
                 -HT_INFINITY);
    HT_ASSERT(ht_compute_variance_reduction(&pre_100, &lhs_4, &rhs_96,
                                            HT_CONST(1.0), HT_CONST(0.03)) > -HT_INFINITY);

    ht_var_stat_t pre_small, l_one, r_one;
    ht_var_init(&pre_small);
    ht_var_init(&l_one);
    ht_var_init(&r_one);
    ht_var_update(&pre_small, HT_CONST(1.0), HT_CONST(1.0));
    ht_var_update(&pre_small, HT_CONST(2.0), HT_CONST(1.0));
    ht_var_update(&l_one, HT_CONST(1.0), HT_CONST(1.0));
    ht_var_update(&r_one, HT_CONST(2.0), HT_CONST(1.0));
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_small, &l_one, &r_one,
                                               HT_CONST(1.0), HT_CONST(0.01)),
                 -HT_INFINITY);

    HT_ASSERT_EQ(ht_compute_variance_reduction(NULL, &lhs_perf, &rhs_perf, HT_CONST(1.0), HT_CONST(0.01)), -HT_INFINITY);
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_var, NULL, &rhs_perf, HT_CONST(1.0), HT_CONST(0.01)), -HT_INFINITY);
    HT_ASSERT_EQ(ht_compute_variance_reduction(&pre_var, &lhs_perf, NULL, HT_CONST(1.0), HT_CONST(0.01)), -HT_INFINITY);
}

static void test_should_split_reg_analytical(void) {
    ht_split_candidate_t best, second_best;
    ht_split_candidate_init(&best);
    ht_split_candidate_init(&second_best);

    ht_float_t tau = HT_CONST(0.05);

    HT_ASSERT(!ht_should_split_reg(&best, &second_best, HT_CONST(0.1), tau));

    best.merit = HT_CONST(0.0);
    best.feature_idx = 0;
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, HT_CONST(0.1), tau));
    best.merit = HT_CONST(-1.5);
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, HT_CONST(0.1), tau));

    best.merit = HT_CONST(10.0);
    best.feature_idx = 0;
    HT_ASSERT(ht_should_split_reg(&best, NULL, HT_CONST(0.5), tau));
    HT_ASSERT(ht_should_split_reg(&best, &second_best, HT_CONST(0.5), tau));

    second_best.feature_idx = 1;
    second_best.merit = HT_CONST(0.0);
    HT_ASSERT(ht_should_split_reg(&best, &second_best, HT_CONST(0.5), tau));
    second_best.merit = HT_CONST(-5.0);
    HT_ASSERT(ht_should_split_reg(&best, &second_best, HT_CONST(0.5), tau));

    second_best.merit = HT_CONST(4.0);
    HT_ASSERT(ht_should_split_reg(&best, &second_best, HT_CONST(0.3), tau));
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, HT_CONST(0.7), tau));

    second_best.merit = HT_CONST(9.8);
    HT_ASSERT(ht_should_split_reg(&best, &second_best, HT_CONST(0.04), tau));
    HT_ASSERT(!ht_should_split_reg(&best, &second_best, HT_CONST(0.06), tau));

    HT_ASSERT(!ht_should_split_reg(NULL, NULL, HT_CONST(0.1), tau));
}

static void test_evaluate_reg_numeric_and_categorical_observers(void) {
    ht_var_stat_t pre_var;
    ht_var_init(&pre_var);
    ht_reg_numeric_obs_t num_obs;
    ht_reg_numeric_init(&num_obs);

    for (int i = 0; i < 15; i++) {
        ht_float_t x_val = HT_FROM_INT(i % 5);
        ht_float_t y_val = HT_CONST(2.0);
        ht_var_update(&pre_var, y_val, HT_CONST(1.0));
        ht_reg_numeric_update(&num_obs, x_val, y_val, HT_CONST(1.0));
    }
    for (int i = 0; i < 15; i++) {
        ht_float_t x_val = HT_FROM_INT(6 + (i % 5));
        ht_float_t y_val = HT_CONST(20.0);
        ht_var_update(&pre_var, y_val, HT_CONST(1.0));
        ht_reg_numeric_update(&num_obs, x_val, y_val, HT_CONST(1.0));
    }

    ht_split_candidate_t num_cand;
    ht_evaluate_reg_numeric_observer(&num_obs, &pre_var, 0, HT_CONST(5.0), HT_CONST(0.01), &num_cand);

    HT_ASSERT(num_cand.merit > HT_CONST(50.0));
    HT_ASSERT_EQ(num_cand.feature_idx, 0);
    HT_ASSERT_EQ(num_cand.is_categorical, 0);
    HT_ASSERT(num_cand.split_val.threshold >= HT_CONST(2.0) &&
              num_cand.split_val.threshold <= HT_CONST(7.0));
    HT_ASSERT_FLOAT_NEAR(num_cand.lhs_var.count, HT_CONST(15.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(num_cand.rhs_var.count, HT_CONST(15.0), HT_TEST_TOL(1e-4));

    ht_var_stat_t cat_pre_var;
    ht_var_init(&cat_pre_var);
    ht_reg_categorical_obs_t cat_obs;
    ht_reg_categorical_init(&cat_obs);

    for (int i = 0; i < 10; i++) {
        ht_var_update(&cat_pre_var, HT_CONST(0.0), HT_CONST(1.0));
        ht_reg_categorical_update(&cat_obs, 0, HT_CONST(0.0), HT_CONST(1.0));
    }
    for (int i = 0; i < 10; i++) {
        ht_var_update(&cat_pre_var, HT_CONST(20.0), HT_CONST(1.0));
        ht_reg_categorical_update(&cat_obs, 1, HT_CONST(20.0), HT_CONST(1.0));
    }
    for (int i = 0; i < 10; i++) {
        ht_var_update(&cat_pre_var, HT_CONST(20.0), HT_CONST(1.0));
        ht_reg_categorical_update(&cat_obs, 2, HT_CONST(20.0), HT_CONST(1.0));
    }

    ht_split_candidate_t cat_cand;
    ht_evaluate_reg_categorical_observer(&cat_obs, &cat_pre_var, 1, HT_CONST(5.0), HT_CONST(0.01), &cat_cand);

    HT_ASSERT(cat_cand.merit > HT_CONST(50.0));
    HT_ASSERT_EQ(cat_cand.feature_idx, 1);
    HT_ASSERT_EQ(cat_cand.is_categorical, 1);
    HT_ASSERT_EQ(cat_cand.split_val.cat_val, 0);
    HT_ASSERT_FLOAT_NEAR(cat_cand.lhs_var.count, HT_CONST(10.0), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(cat_cand.rhs_var.count, HT_CONST(20.0), HT_TEST_TOL(1e-4));
}

static void test_find_best_reg_split_candidates_and_tie_breaking(void) {
    static ht_memory_pool_t pool;
    ht_memory_init(&pool);

    uint16_t stats_idx = ht_stats_alloc(&pool);
    HT_ASSERT_NEQ(stats_idx, HT_INVALID_IDX);
    ht_leaf_stats_t *stats = ht_stats_get(&pool, stats_idx);
    HT_ASSERT_NOT_NULL(stats);

    ht_feature_type_t feat_types[HT_MAX_FEATURES];
    feat_types[0] = HT_FEAT_NUMERIC;
    feat_types[1] = HT_FEAT_NUMERIC;
    feat_types[2] = HT_FEAT_CATEGORICAL;
    feat_types[3] = HT_FEAT_NUMERIC;

    ht_var_stat_t pre_var;
    ht_var_init(&pre_var);

    for (uint8_t f = 0; f < 4; f++) {
        if (feat_types[f] == HT_FEAT_NUMERIC) {
            ht_reg_numeric_init(&stats->observers[f].reg_num);
        } else {
            ht_reg_categorical_init(&stats->observers[f].reg_cat);
        }
    }

    for (int i = 0; i < 40; i++) {
        ht_float_t x0 = (i < 20) ? HT_CONST(2.0) : HT_CONST(8.0);
        ht_float_t x1 = (i < 8)  ? HT_CONST(1.0) : HT_CONST(9.0);
        int32_t    x2 = (i < 15) ? 0 : 1;
        ht_float_t x3 = HT_CONST(5.0);
        ht_float_t y  = (i < 20) ? HT_CONST(10.0) : HT_CONST(50.0);

        ht_var_update(&pre_var, y, HT_CONST(1.0));
        ht_reg_numeric_update(&stats->observers[0].reg_num, x0, y, HT_CONST(1.0));
        ht_reg_numeric_update(&stats->observers[1].reg_num, x1, y, HT_CONST(1.0));
        ht_reg_categorical_update(&stats->observers[2].reg_cat, x2, y, HT_CONST(1.0));
        ht_reg_numeric_update(&stats->observers[3].reg_num, x3, y, HT_CONST(1.0));
    }

    ht_split_candidate_t best, second_best;
    ht_find_best_reg_split_candidates(stats, &pre_var, feat_types, 4,
                                      HT_CONST(5.0), HT_CONST(0.01), true,
                                      &best, &second_best);

    HT_ASSERT_EQ(best.feature_idx, 0);
    HT_ASSERT(best.merit > HT_CONST(100.0));
    HT_ASSERT(second_best.feature_idx != HT_NO_FEATURE);
    HT_ASSERT(second_best.merit > HT_CONST(0.0));
    HT_ASSERT(second_best.merit <= best.merit);

    ht_leaf_stats_t *tie_stats = ht_stats_get(&pool, stats_idx);
    ht_reg_numeric_init(&tie_stats->observers[0].reg_num);
    ht_reg_numeric_init(&tie_stats->observers[1].reg_num);

    for (int i = 0; i < 20; i++) {
        ht_reg_numeric_update(&tie_stats->observers[0].reg_num, HT_CONST(1.0), HT_CONST(5.0), HT_CONST(1.0));
        ht_reg_numeric_update(&tie_stats->observers[1].reg_num, HT_CONST(1.0), HT_CONST(5.0), HT_CONST(1.0));
    }
    for (int i = 0; i < 20; i++) {
        ht_reg_numeric_update(&tie_stats->observers[0].reg_num, HT_CONST(9.0), HT_CONST(45.0), HT_CONST(1.0));
        ht_reg_numeric_update(&tie_stats->observers[1].reg_num, HT_CONST(9.0), HT_CONST(45.0), HT_CONST(1.0));
    }

    ht_feature_type_t tie_types[HT_MAX_FEATURES];
    tie_types[0] = HT_FEAT_NUMERIC;
    tie_types[1] = HT_FEAT_NUMERIC;
    ht_find_best_reg_split_candidates(tie_stats, &pre_var, tie_types, 2,
                                      HT_CONST(5.0), HT_CONST(0.01), true,
                                      &best, &second_best);

    HT_ASSERT_FLOAT_NEAR(best.merit, second_best.merit, HT_TEST_TOL(1e-5));
    HT_ASSERT_EQ(best.feature_idx, 1);
    HT_ASSERT_EQ(second_best.feature_idx, 0);

    ht_stats_free(&pool, stats_idx);
}

static void test_reg_criterion_vtable_and_null_edge_cases(void) {
    const ht_criterion_vtable_t *crit = ht_criterion_get_vtable(HT_CRIT_VARIANCE_REDUCTION);
    HT_ASSERT_NOT_NULL(crit);
    ht_float_t dummy_dist[HT_MAX_CLASSES] = { HT_CONST(10.0), HT_CONST(10.0), 0, 0 };
    HT_ASSERT_FLOAT_NEAR(crit->range_of_merit(dummy_dist, 2), HT_CONST(1.0), HT_TEST_TOL(1e-6));
    HT_ASSERT_EQ(crit->merit_of_split(dummy_dist, dummy_dist, dummy_dist, 2, HT_CONST(0.01)), -HT_INFINITY);

    ht_split_candidate_t cand;
    ht_evaluate_reg_numeric_observer(NULL, NULL, 0, HT_CONST(1.0), HT_CONST(0.01), &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_evaluate_reg_categorical_observer(NULL, NULL, 0, HT_CONST(1.0), HT_CONST(0.01), &cand);
    HT_ASSERT_EQ(cand.merit, -HT_INFINITY);

    ht_split_candidate_t best, second;
    ht_find_best_reg_split_candidates(NULL, NULL, NULL, 0, HT_CONST(1.0), HT_CONST(0.01), true, &best, &second);
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
