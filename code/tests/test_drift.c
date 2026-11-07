/**
 * @file test_drift.c
 * @brief Unit, Integration, and Parity Tests for ADWIN Concept Drift Detector and Pool.
 *
 * Covers Level 1 (Unit: math, buckets, compression, cuts, edge cases),
 * and Level 2 (Integration: River parity, step change, multi-drift, and static pool lifecycle).
 */

#include "test_harness.h"
#include "ht_drift.h"
#include "ht_tree.h"

/* --- Level 1: Unit Tests --- */

static void test_adwin_default_config_and_init(void) {
    ht_adwin_config_t cfg;
    ht_adwin_default_config(&cfg);

    HT_ASSERT_FLOAT_NEAR(cfg.delta, (ht_float_t)0.002, (ht_float_t)1e-6);
    HT_ASSERT_EQ(cfg.clock, 32U);
    HT_ASSERT_EQ(cfg.min_window_length, 5U);
    HT_ASSERT_EQ(cfg.grace_period, 10U);
    HT_ASSERT_EQ(cfg.max_buckets, 5U);

    /* Valid initialization with defaults */
    ht_adwin_t ad;
    HT_ASSERT_EQ(ht_adwin_init(&ad, NULL), HT_OK);
    HT_ASSERT_EQ(ht_adwin_get_width(&ad), 0U);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_total(&ad), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_variance(&ad), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_estimation(&ad), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_adwin_get_n_detections(&ad), 0U);
    HT_ASSERT_EQ(ad.last_bucket_row, 0U);
    HT_ASSERT(!ad.drift_detected);

    /* Invalid arguments rejection */
    HT_ASSERT_EQ(ht_adwin_init(NULL, NULL), HT_ERR_INVALID_ARG);

    ht_adwin_config_t bad_cfg = cfg;
    bad_cfg.delta = (ht_float_t)0.0; /* delta must be > 0 */
    HT_ASSERT_EQ(ht_adwin_init(&ad, &bad_cfg), HT_ERR_INVALID_ARG);

    bad_cfg = cfg;
    bad_cfg.delta = (ht_float_t)1.0; /* delta must be < 1 */
    HT_ASSERT_EQ(ht_adwin_init(&ad, &bad_cfg), HT_ERR_INVALID_ARG);

    bad_cfg = cfg;
    bad_cfg.clock = 0U; /* clock must be > 0 */
    HT_ASSERT_EQ(ht_adwin_init(&ad, &bad_cfg), HT_ERR_INVALID_ARG);

    bad_cfg = cfg;
    bad_cfg.min_window_length = 0U;
    HT_ASSERT_EQ(ht_adwin_init(&ad, &bad_cfg), HT_ERR_INVALID_ARG);

    bad_cfg = cfg;
    bad_cfg.max_buckets = 6U; /* exceeds HT_ADWIN_MAX_BUCKETS = 5 */
    HT_ASSERT_EQ(ht_adwin_init(&ad, &bad_cfg), HT_ERR_INVALID_ARG);
}

static void test_adwin_single_and_incremental_stats(void) {
    ht_adwin_t ad;
    HT_ASSERT_EQ(ht_adwin_init(&ad, NULL), HT_OK);

    /* Insert 1st element: 5.0 */
    bool drift = ht_adwin_update(&ad, (ht_float_t)5.0);
    HT_ASSERT(!drift);
    HT_ASSERT_EQ(ht_adwin_get_width(&ad), 1U);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_total(&ad), (ht_float_t)5.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_estimation(&ad), (ht_float_t)5.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_variance(&ad), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_variance_in_window(&ad), (ht_float_t)0.0);

    /* Insert 2nd element: 15.0
     * n = 2, total = 20.0, mean = 10.0
     * M2 = (5-10)^2 + (15-10)^2 = 25 + 25 = 50.0
     * var_in_window = 50.0 / 2 = 25.0
     */
    drift = ht_adwin_update(&ad, (ht_float_t)15.0);
    HT_ASSERT(!drift);
    HT_ASSERT_EQ(ht_adwin_get_width(&ad), 2U);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_total(&ad), (ht_float_t)20.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_estimation(&ad), (ht_float_t)10.0);
    HT_ASSERT_FLOAT_NEAR(ht_adwin_get_variance(&ad), (ht_float_t)50.0, (ht_float_t)1e-4);
    HT_ASSERT_FLOAT_NEAR(ht_adwin_get_variance_in_window(&ad), (ht_float_t)25.0, (ht_float_t)1e-4);

    /* Insert 3rd element: 10.0
     * n = 3, total = 30.0, mean = 10.0
     * M2 = 50.0 + (10 - 10)^2 = 50.0
     * var_in_window = 50.0 / 3 = 16.6667
     */
    drift = ht_adwin_update(&ad, (ht_float_t)10.0);
    HT_ASSERT(!drift);
    HT_ASSERT_EQ(ht_adwin_get_width(&ad), 3U);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_total(&ad), (ht_float_t)30.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_estimation(&ad), (ht_float_t)10.0);
    HT_ASSERT_FLOAT_NEAR(ht_adwin_get_variance(&ad), (ht_float_t)50.0, (ht_float_t)1e-4);
    HT_ASSERT_FLOAT_NEAR(ht_adwin_get_variance_in_window(&ad), (ht_float_t)(50.0 / 3.0), (ht_float_t)1e-4);
}

static void test_adwin_bucket_compression_powers_of_two(void) {
    ht_adwin_t ad;
    HT_ASSERT_EQ(ht_adwin_init(&ad, NULL), HT_OK);

    /* Insert 5 elements: 1, 2, 3, 4, 5
     * max_buckets = 5: all fit in row 0 without compression
     */
    for (int i = 1; i <= 5; ++i) {
        ht_adwin_update(&ad, (ht_float_t)i);
    }
    HT_ASSERT_EQ(ad.rows[0].count, 5U);
    HT_ASSERT_EQ(ad.rows[1].count, 0U);
    HT_ASSERT_EQ(ad.last_bucket_row, 0U);

    /* Insert 6th element: triggers compression!
     * Oldest two buckets in row 0 (1 and 2) are merged into row 1.
     * Merged bucket in row 1: sum = 1 + 2 = 3.0, capacity = 2^1 = 2.
     * Row 0 count becomes 5 + 1 - 2 = 4.
     * Row 1 count becomes 1.
     */
    ht_adwin_update(&ad, (ht_float_t)6.0);
    HT_ASSERT_EQ(ad.rows[0].count, 4U);
    HT_ASSERT_EQ(ad.rows[1].count, 1U);
    HT_ASSERT_FLOAT_EQ(ad.rows[1].sum[0], (ht_float_t)3.0);
    HT_ASSERT_EQ(ad.last_bucket_row, 1U);

    /* Insert more elements to trigger cascading into level 2 (capacity 2^2 = 4) */
    for (int i = 7; i <= 16; ++i) {
        ht_adwin_update(&ad, (ht_float_t)i);
    }

    /* Total elements = 16, total sum = 16*17/2 = 136 */
    HT_ASSERT_EQ(ht_adwin_get_width(&ad), 16U);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_total(&ad), (ht_float_t)136.0);

    /* Verify sum of all active buckets equals total_sum */
    ht_float_t sum_all_buckets = (ht_float_t)0.0;
    for (uint8_t r = 0; r < HT_ADWIN_LEVELS; ++r) {
        for (uint8_t b = 0; b < ad.rows[r].count; ++b) {
            sum_all_buckets += ad.rows[r].sum[b];
        }
    }
    HT_ASSERT_FLOAT_NEAR(sum_all_buckets, (ht_float_t)136.0, (ht_float_t)1e-4);
}

static void test_adwin_cut_epsilon_analytical(void) {
    /* Hand calculated analytical verification:
     * n = 100, n0 = 50, n1 = 50, var = 0.25, delta = 0.002, min_w = 5
     * ln(100) = 4.60517
     * 2 * ln(100) / delta = 4605.17
     * dd = ln(4605.17) = 8.43493
     * m = 1/(50-5+1) + 1/(50-5+1) = 2/46 = 0.043478
     * term1 = sqrt(2 * 0.043478 * 0.25 * 8.43493) = sqrt(0.183368) = 0.428215
     * term2 = (2/3) * 8.43493 * 0.043478 = 0.244491
     * epsilon = 0.428215 + 0.244491 = 0.672706
     */
    ht_float_t eps = ht_adwin_calculate_cut_epsilon(100U, 50U, 50U, (ht_float_t)0.25, (ht_float_t)0.002, 5U);
    HT_ASSERT_FLOAT_NEAR(eps, (ht_float_t)0.672706, (ht_float_t)1e-3);

    /* Edge cases returning HT_INFINITY */
    HT_ASSERT_FLOAT_EQ(ht_adwin_calculate_cut_epsilon(0U, 0U, 0U, (ht_float_t)0.25, (ht_float_t)0.002, 5U), HT_INFINITY);
    HT_ASSERT_FLOAT_EQ(ht_adwin_calculate_cut_epsilon(1U, 0U, 1U, (ht_float_t)0.25, (ht_float_t)0.002, 5U), HT_INFINITY);
    HT_ASSERT_FLOAT_EQ(ht_adwin_calculate_cut_epsilon(100U, 4U, 96U, (ht_float_t)0.25, (ht_float_t)0.002, 5U), HT_INFINITY);
    HT_ASSERT_FLOAT_EQ(ht_adwin_calculate_cut_epsilon(100U, 50U, 50U, (ht_float_t)0.25, (ht_float_t)0.0, 5U), HT_INFINITY);
}

static void test_adwin_edge_cases_and_null(void) {
    ht_adwin_t ad;
    HT_ASSERT_EQ(ht_adwin_init(&ad, NULL), HT_OK);

    /* NULL detector protection */
    HT_ASSERT(!ht_adwin_update(NULL, (ht_float_t)1.0));
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_estimation(NULL), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_variance(NULL), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_variance_in_window(NULL), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_total(NULL), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_adwin_get_width(NULL), 0U);
    HT_ASSERT_EQ(ht_adwin_get_n_detections(NULL), 0U);
    ht_adwin_reset(NULL);

    /* NaN and Infinity inputs must be safely ignored */
    ht_float_t zero = (ht_float_t)0.0;
    ht_float_t nan_val = zero / zero;
    HT_ASSERT(!ht_adwin_update(&ad, nan_val));
    HT_ASSERT(!ht_adwin_update(&ad, HT_INFINITY));
    HT_ASSERT(!ht_adwin_update(&ad, -HT_INFINITY));
    HT_ASSERT_EQ(ht_adwin_get_width(&ad), 0U);

    /* Stationary constant stream: 1000 identical items -> zero false positives */
    for (int i = 0; i < 1000; ++i) {
        bool drift = ht_adwin_update(&ad, (ht_float_t)42.0);
        HT_ASSERT(!drift);
    }
    HT_ASSERT_EQ(ht_adwin_get_n_detections(&ad), 0U);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_estimation(&ad), (ht_float_t)42.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_variance_in_window(&ad), (ht_float_t)0.0);

    /* Reset test */
    ht_adwin_reset(&ad);
    HT_ASSERT_EQ(ht_adwin_get_width(&ad), 0U);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_total(&ad), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_adwin_get_variance(&ad), (ht_float_t)0.0);
    HT_ASSERT_EQ(ht_adwin_get_n_detections(&ad), 0U);
    HT_ASSERT_EQ(ad.delta, (ht_float_t)0.002);
    HT_ASSERT_EQ(ad.clock, 32U);
}

/* --- Level 2: Integration & River Parity Tests --- */

static void test_adwin_drift_detection_step_change_river_parity(void) {
    ht_adwin_t ad;
    HT_ASSERT_EQ(ht_adwin_init(&ad, NULL), HT_OK);

    /* Feed 100 zeros followed by 100 ones
     * Exactly matches River's AdaptiveWindowing engine:
     * Detections must occur at:
     * - Step 127: width=56, total=28.0, var=14.0, mean=0.5
     * - Step 159: width=72, total=60.0, var=10.0, mean=0.8333
     */
    uint32_t drift_step_1 = 0U;
    uint32_t drift_step_2 = 0U;

    for (uint32_t i = 0; i < 200U; ++i) {
        ht_float_t val = (i < 100U) ? (ht_float_t)0.0 : (ht_float_t)1.0;
        bool drift = ht_adwin_update(&ad, val);

        if (drift) {
            if (drift_step_1 == 0U) {
                drift_step_1 = i;
                /* Exact parity check on step 127 */
                HT_ASSERT_EQ(i, 127U);
                HT_ASSERT_EQ(ht_adwin_get_width(&ad), 56U);
                HT_ASSERT_FLOAT_NEAR(ht_adwin_get_total(&ad), (ht_float_t)28.0, (ht_float_t)1e-4);
                HT_ASSERT_FLOAT_NEAR(ht_adwin_get_estimation(&ad), (ht_float_t)0.5, (ht_float_t)1e-4);
                HT_ASSERT_FLOAT_NEAR(ht_adwin_get_variance(&ad), (ht_float_t)14.0, (ht_float_t)1e-4);
                HT_ASSERT_FLOAT_NEAR(ht_adwin_get_variance_in_window(&ad), (ht_float_t)0.25, (ht_float_t)1e-4);
                HT_ASSERT_EQ(ht_adwin_get_n_detections(&ad), 1U);
            } else if (drift_step_2 == 0U) {
                drift_step_2 = i;
                /* Exact parity check on step 159 */
                HT_ASSERT_EQ(i, 159U);
                HT_ASSERT_EQ(ht_adwin_get_width(&ad), 72U);
                HT_ASSERT_FLOAT_NEAR(ht_adwin_get_total(&ad), (ht_float_t)60.0, (ht_float_t)1e-4);
                HT_ASSERT_FLOAT_NEAR(ht_adwin_get_variance(&ad), (ht_float_t)10.0, (ht_float_t)1e-4);
                HT_ASSERT_EQ(ht_adwin_get_n_detections(&ad), 2U);
            }
        }
    }

    HT_ASSERT_EQ(drift_step_1, 127U);
    HT_ASSERT_EQ(drift_step_2, 159U);
}

static void test_adwin_drift_detection_multiple_shifts(void) {
    ht_adwin_t ad;
    HT_ASSERT_EQ(ht_adwin_init(&ad, NULL), HT_OK);

    /* Stream with 3 distinct distribution shifts:
     * 0..299: mean 0.0
     * 300..599: mean 10.0 (abrupt shift up)
     * 600..899: mean 2.0 (abrupt shift down)
     */
    uint32_t detections_in_phase2 = 0U;
    uint32_t detections_in_phase3 = 0U;

    for (uint32_t i = 0; i < 900U; ++i) {
        ht_float_t val;
        if (i < 300U) {
            val = (ht_float_t)0.0;
        } else if (i < 600U) {
            val = (ht_float_t)10.0;
        } else {
            val = (ht_float_t)2.0;
        }

        bool drift = ht_adwin_update(&ad, val);
        if (drift) {
            if (i >= 300U && i < 600U) {
                detections_in_phase2++;
            } else if (i >= 600U) {
                detections_in_phase3++;
            }
        }
    }

    /* Both abrupt shifts must have triggered drift detections */
    HT_ASSERT(detections_in_phase2 > 0U);
    HT_ASSERT(detections_in_phase3 > 0U);
    HT_ASSERT_EQ(ht_adwin_get_n_detections(&ad), detections_in_phase2 + detections_in_phase3);

    /* In the final stationary regime, mean must have adapted near 2.0 */
    HT_ASSERT_FLOAT_NEAR(ht_adwin_get_estimation(&ad), (ht_float_t)2.0, (ht_float_t)0.1);
}

static void test_adwin_pool_alloc_free_lifecycle(void) {
    ht_adwin_pool_t pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&pool), HT_OK);
    HT_ASSERT_EQ(pool.free_top, HT_MAX_ADWIN_DETECTORS);
    HT_ASSERT_EQ(pool.allocated_count, 0U);

    /* Allocate all detectors up to capacity */
    uint16_t indices[HT_MAX_ADWIN_DETECTORS];
    for (uint16_t i = 0; i < HT_MAX_ADWIN_DETECTORS; ++i) {
        indices[i] = ht_adwin_pool_alloc(&pool, NULL);
        HT_ASSERT_NEQ(indices[i], HT_INVALID_IDX);
        HT_ASSERT(indices[i] < HT_MAX_ADWIN_DETECTORS);

        ht_adwin_t *ad = ht_adwin_pool_get(&pool, indices[i]);
        HT_ASSERT_NOT_NULL(ad);
        HT_ASSERT_EQ(ad->width, 0U);
    }

    HT_ASSERT_EQ(pool.allocated_count, HT_MAX_ADWIN_DETECTORS);
    HT_ASSERT_EQ(pool.free_top, 0U);

    /* Next allocation must fail */
    HT_ASSERT_EQ(ht_adwin_pool_alloc(&pool, NULL), HT_INVALID_IDX);

    /* Free detector at index 5 and index 10 */
    HT_ASSERT_EQ(ht_adwin_pool_free(&pool, indices[5]), HT_OK);
    HT_ASSERT_EQ(ht_adwin_pool_free(&pool, indices[10]), HT_OK);
    HT_ASSERT_EQ(pool.allocated_count, (uint16_t)(HT_MAX_ADWIN_DETECTORS - 2U));

    /* Access to freed detector must return NULL */
    HT_ASSERT_NULL(ht_adwin_pool_get(&pool, indices[5]));
    HT_ASSERT_NULL(ht_adwin_pool_get(&pool, indices[10]));

    /* Double-free must fail with error */
    HT_ASSERT_EQ(ht_adwin_pool_free(&pool, indices[5]), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_adwin_pool_free(&pool, indices[10]), HT_ERR_INVALID_ARG);

    /* Out of bounds free and get */
    HT_ASSERT_EQ(ht_adwin_pool_free(&pool, HT_MAX_ADWIN_DETECTORS), HT_ERR_INVALID_ARG);
    HT_ASSERT_NULL(ht_adwin_pool_get(&pool, HT_MAX_ADWIN_DETECTORS));

    /* Re-allocate the 2 freed slots */
    uint16_t realloc1 = ht_adwin_pool_alloc(&pool, NULL);
    uint16_t realloc2 = ht_adwin_pool_alloc(&pool, NULL);
    HT_ASSERT_NEQ(realloc1, HT_INVALID_IDX);
    HT_ASSERT_NEQ(realloc2, HT_INVALID_IDX);
    HT_ASSERT_EQ(pool.allocated_count, HT_MAX_ADWIN_DETECTORS);

    /* Reset pool */
    ht_adwin_pool_reset(&pool);
    HT_ASSERT_EQ(pool.allocated_count, 0U);
    HT_ASSERT_EQ(pool.free_top, HT_MAX_ADWIN_DETECTORS);
}

static void test_adwin_pool_concurrent_detectors(void) {
    ht_adwin_pool_t pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&pool), HT_OK);

    /* Allocate 2 detectors with different delta values */
    ht_adwin_config_t cfg_sensitive;
    ht_adwin_default_config(&cfg_sensitive);
    cfg_sensitive.delta = (ht_float_t)0.01; /* more sensitive */

    ht_adwin_config_t cfg_robust;
    ht_adwin_default_config(&cfg_robust);
    cfg_robust.delta = (ht_float_t)0.0001; /* more conservative */

    uint16_t idx_sens = ht_adwin_pool_alloc(&pool, &cfg_sensitive);
    uint16_t idx_rob = ht_adwin_pool_alloc(&pool, &cfg_robust);

    ht_adwin_t *ad_sens = ht_adwin_pool_get(&pool, idx_sens);
    ht_adwin_t *ad_rob = ht_adwin_pool_get(&pool, idx_rob);
    HT_ASSERT_NOT_NULL(ad_sens);
    HT_ASSERT_NOT_NULL(ad_rob);

    /* Feed both detectors the same stream: 100 zeros then 100 ones */
    uint32_t sens_first_drift = 0U;
    uint32_t rob_first_drift = 0U;

    for (uint32_t i = 0; i < 200U; ++i) {
        ht_float_t val = (i < 100U) ? (ht_float_t)0.0 : (ht_float_t)1.0;
        if (ht_adwin_update(ad_sens, val) && sens_first_drift == 0U) {
            sens_first_drift = i;
        }
        if (ht_adwin_update(ad_rob, val) && rob_first_drift == 0U) {
            rob_first_drift = i;
        }
    }

    /* Sensitive detector must detect drift at least as early as conservative detector */
    HT_ASSERT(sens_first_drift > 0U);
    HT_ASSERT(rob_first_drift > 0U);
    HT_ASSERT(sens_first_drift <= rob_first_drift);

    /* Clean up */
    HT_ASSERT_EQ(ht_adwin_pool_free(&pool, idx_sens), HT_OK);
    HT_ASSERT_EQ(ht_adwin_pool_free(&pool, idx_rob), HT_OK);
}

static void test_adwin_pool_stress_50k_recycling(void) {
    ht_adwin_pool_t pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&pool), HT_OK);

    uint16_t active[16];
    for (int i = 0; i < 16; ++i) {
        active[i] = ht_adwin_pool_alloc(&pool, NULL);
        HT_ASSERT_NEQ(active[i], HT_INVALID_IDX);
    }

    /* 50,000 rapid recycling cycles */
    uint32_t lcg = 123456789U;
    for (int iter = 0; iter < 50000; ++iter) {
        lcg = lcg * 1664525U + 1013904223U;
        int slot = (int)(lcg % 16U);

        /* Free slot */
        HT_ASSERT_EQ(ht_adwin_pool_free(&pool, active[slot]), HT_OK);

        /* Re-allocate slot */
        active[slot] = ht_adwin_pool_alloc(&pool, NULL);
        HT_ASSERT_NEQ(active[slot], HT_INVALID_IDX);

        /* Update with sample */
        ht_adwin_t *ad = ht_adwin_pool_get(&pool, active[slot]);
        HT_ASSERT_NOT_NULL(ad);
        ht_adwin_update(ad, (ht_float_t)(iter % 10));
    }

    HT_ASSERT_EQ(pool.allocated_count, 16U);
}

static void test_adwin_pool_bytes_budget(void) {
    ht_adwin_pool_t pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&pool), HT_OK);

    size_t bytes = ht_adwin_pool_bytes_used(&pool);
    /* Budget: <= 256 KB */
    HT_ASSERT(bytes <= (256U * 1024U));
    HT_ASSERT_EQ(bytes, sizeof(ht_adwin_pool_t));
}

/* --- Level 1: HAT Unit Tests (Filter, Bound, Voting) --- */

static void test_hat_unilateral_error_filter(void) {
    ht_adwin_t adwin;
    HT_ASSERT_EQ(ht_adwin_init(&adwin, NULL), HT_OK);

    ht_float_t mean_err = (ht_float_t)0.0;
    uint32_t err_count = 0;

    /* Case A: Null checks */
    HT_ASSERT(!ht_hat_check_error_drift(NULL, &mean_err, &err_count, 1));
    HT_ASSERT(!ht_hat_check_error_drift(&adwin, NULL, &err_count, 1));
    HT_ASSERT(!ht_hat_check_error_drift(&adwin, &mean_err, NULL, 1));

    /* Case B: Error is decreasing (improving accuracy)
     * Start with high error: 40 samples of 1 (error rate 1.0)
     */
    for (int i = 0; i < 40; ++i) {
        ht_hat_check_error_drift(&adwin, &mean_err, &err_count, 1);
    }
    HT_ASSERT(mean_err > (ht_float_t)0.8);

    /* Now feed 100 accurate predictions (0s): error drops sharply.
     * When ADWIN triggers drift, it represents error decreasing!
     */
    bool decreasing_drift_reported = false;
    for (int i = 0; i < 100; ++i) {
        if (ht_hat_check_error_drift(&adwin, &mean_err, &err_count, 0)) {
            decreasing_drift_reported = true;
        }
    }
    /* The unilateral filter must suppress the drift because error decreased! */
    HT_ASSERT(!decreasing_drift_reported);

    /* Case C: Error is increasing (degrading accuracy)
     * Start with low error: 40 samples of 0 (error rate ~0.0)
     */
    ht_adwin_reset(&adwin);
    mean_err = (ht_float_t)0.0;
    err_count = 0;

    for (int i = 0; i < 40; ++i) {
        ht_hat_check_error_drift(&adwin, &mean_err, &err_count, 0);
    }
    HT_ASSERT_FLOAT_NEAR(mean_err, (ht_float_t)0.0, 1e-5);

    /* Now feed 100 errors (1s): error rises sharply.
     * When ADWIN triggers drift, error is increasing!
     */
    bool increasing_drift_reported = false;
    for (int i = 0; i < 100; ++i) {
        if (ht_hat_check_error_drift(&adwin, &mean_err, &err_count, 1)) {
            increasing_drift_reported = true;
        }
    }
    /* The unilateral filter must report genuine drift! */
    HT_ASSERT(increasing_drift_reported);
}

static void test_hat_switch_prune_bound_analytical(void) {
    /* Exact analytical test:
     * e_old = 0.40, n_old = 400, n_alt = 400, alpha = 0.05
     * var = 0.40 * 0.60 = 0.24
     * ln(2 / 0.05) = ln(40) = 3.6888794541
     * 1/400 + 1/400 = 0.005
     * 2 * 0.24 * 3.6888794541 * 0.005 = 0.0088533107
     * bound = sqrt(0.0088533107) = 0.09409203
     */
    ht_float_t bound = ht_hat_calculate_switch_bound(
        (ht_float_t)0.40, 400U, 400U, (ht_float_t)0.05
    );
    HT_ASSERT_FLOAT_NEAR(bound, (ht_float_t)0.09409203, 1e-4);

    /* Test equal sample sizes with different error */
    /* e_old = 0.50, n_old = 500, n_alt = 500, alpha = 0.05 */
    /* var = 0.25, ln(40) = 3.688879, inv_n = 0.004 */
    /* bound = sqrt(2 * 0.25 * 3.688879 * 0.004) = sqrt(0.00737776) = 0.0858939 */
    bound = ht_hat_calculate_switch_bound(
        (ht_float_t)0.50, 500U, 500U, (ht_float_t)0.05
    );
    HT_ASSERT_FLOAT_NEAR(bound, (ht_float_t)0.0858939, 1e-4);

    /* Edge cases */
    HT_ASSERT_EQ(ht_hat_calculate_switch_bound((ht_float_t)0.5, 0U, 500U, (ht_float_t)0.05), HT_INFINITY);
    HT_ASSERT_EQ(ht_hat_calculate_switch_bound((ht_float_t)0.5, 500U, 0U, (ht_float_t)0.05), HT_INFINITY);
    HT_ASSERT_EQ(ht_hat_calculate_switch_bound((ht_float_t)0.5, 500U, 500U, (ht_float_t)0.0), HT_INFINITY);
    HT_ASSERT_EQ(ht_hat_calculate_switch_bound((ht_float_t)0.5, 500U, 500U, (ht_float_t)1.0), HT_INFINITY);

    /* Extreme errors e = 0.0 and e = 1.0 yield bound = 0.0 */
    HT_ASSERT_FLOAT_EQ(ht_hat_calculate_switch_bound((ht_float_t)0.0, 500U, 500U, (ht_float_t)0.05), (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(ht_hat_calculate_switch_bound((ht_float_t)1.0, 500U, 500U, (ht_float_t)0.05), (ht_float_t)0.0);
}

static void test_hat_evaluate_alternate_switch_and_prune(void) {
    /* 1. min_obs rejection: if observations < threshold (300), action must be NONE */
    ht_hat_action_t act = ht_hat_evaluate_alternate(
        (ht_float_t)0.60, 250U, (ht_float_t)0.10, 250U, 300U, (ht_float_t)0.05
    );
    HT_ASSERT_EQ(act, HT_HAT_ACTION_NONE);

    /* 2. SWITCH condition:
     * n_old = 500, n_alt = 500, min_obs = 300, alpha = 0.05 -> bound ~ 0.0859
     * e_old = 0.50, e_alt = 0.20 -> diff = 0.30 > bound
     */
    act = ht_hat_evaluate_alternate(
        (ht_float_t)0.50, 500U, (ht_float_t)0.20, 500U, 300U, (ht_float_t)0.05
    );
    HT_ASSERT_EQ(act, HT_HAT_ACTION_SWITCH);

    /* 3. PRUNE condition:
     * e_old = 0.20, e_alt = 0.50 -> diff e_alt - e_old = 0.30 > bound
     */
    act = ht_hat_evaluate_alternate(
        (ht_float_t)0.20, 500U, (ht_float_t)0.50, 500U, 300U, (ht_float_t)0.05
    );
    HT_ASSERT_EQ(act, HT_HAT_ACTION_PRUNE);

    /* 4. KEEP condition:
     * e_old = 0.35, e_alt = 0.33 -> diff = 0.02 < bound ~ 0.088
     */
    act = ht_hat_evaluate_alternate(
        (ht_float_t)0.35, 500U, (ht_float_t)0.33, 500U, 300U, (ht_float_t)0.05
    );
    HT_ASSERT_EQ(act, HT_HAT_ACTION_NONE);

    /* 5. Zero error in old tree never switches (bound = 0.0 < 0 - e_alt is false) */
    act = ht_hat_evaluate_alternate(
        (ht_float_t)0.0, 500U, (ht_float_t)0.10, 500U, 300U, (ht_float_t)0.05
    );
    HT_ASSERT_EQ(act, HT_HAT_ACTION_PRUNE);
}

static void test_hat_weighted_voting_one_over_error_squared(void) {
    /* 1. Single leaf: probability must be identical to base */
    ht_float_t leaf1_p[HT_MAX_CLASSES] = {(ht_float_t)0.7, (ht_float_t)0.3, (ht_float_t)0.0, (ht_float_t)0.0};
    ht_float_t leaf1_err = (ht_float_t)0.25;
    ht_float_t out[HT_MAX_CLASSES];

    ht_hat_combine_leaf_predictions(leaf1_p, &leaf1_err, 1U, 2U, out);
    HT_ASSERT_FLOAT_NEAR(out[0], (ht_float_t)0.7, 1e-5);
    HT_ASSERT_FLOAT_NEAR(out[1], (ht_float_t)0.3, 1e-5);

    /* 2. Two leaves:
     * Leaf 0: err = 0.50 -> weight = 1 / 0.5^2 = 4.0. Proba = [0.8, 0.2]
     * Leaf 1: err = 0.10 -> weight = 1 / 0.1^2 = 100.0. Proba = [0.1, 0.9]
     * Class 0: 4 * 0.8 + 100 * 0.1 = 3.2 + 10 = 13.2
     * Class 1: 4 * 0.2 + 100 * 0.9 = 0.8 + 90 = 90.8
     * Total = 104.0
     * Normalized: P(0) = 13.2 / 104.0 = 0.126923, P(1) = 90.8 / 104.0 = 0.873077
     */
    ht_float_t two_leaves_p[2 * HT_MAX_CLASSES];
    two_leaves_p[0] = (ht_float_t)0.8; two_leaves_p[1] = (ht_float_t)0.2;
    two_leaves_p[2] = (ht_float_t)0.0; two_leaves_p[3] = (ht_float_t)0.0;

    two_leaves_p[4] = (ht_float_t)0.1; two_leaves_p[5] = (ht_float_t)0.9;
    two_leaves_p[6] = (ht_float_t)0.0; two_leaves_p[7] = (ht_float_t)0.0;

    ht_float_t errs[2] = {(ht_float_t)0.50, (ht_float_t)0.10};
    ht_hat_combine_leaf_predictions(two_leaves_p, errs, 2U, 2U, out);

    HT_ASSERT_FLOAT_NEAR(out[0], (ht_float_t)0.126923, 1e-4);
    HT_ASSERT_FLOAT_NEAR(out[1], (ht_float_t)0.873077, 1e-4);

    /* 3. Zero error floor protection */
    errs[1] = (ht_float_t)0.0; /* 0.0 error */
    ht_hat_combine_leaf_predictions(two_leaves_p, errs, 2U, 2U, out);
    /* Leaf 1 with 0.0 error gets massive weight 1e12 and dominates */
    HT_ASSERT_FLOAT_NEAR(out[0], (ht_float_t)0.1, 1e-3);
    HT_ASSERT_FLOAT_NEAR(out[1], (ht_float_t)0.9, 1e-3);
}

/* --- Level 2: HAT Subtree Replacement and Pruning Integration Tests --- */

static void test_hat_subtree_replacement_and_iterative_recycling(void) {
    ht_adwin_pool_t ad_pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&ad_pool), HT_OK);

    ht_config_params_t params;
    ht_default_hat_params(&params, 4, 2);

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init_hat(&tree, &params, &ad_pool), HT_OK);

    /* Initially 1 node (root), 1 stats, 1 adwin detector */
    HT_ASSERT_EQ(tree.memory.nodes_allocated, 1U);
    HT_ASSERT_EQ(tree.memory.stats_allocated, 1U);
    HT_ASSERT_EQ(ad_pool.allocated_count, 1U);

    /* Split root into left and right child */
    uint16_t root_idx = tree.root_idx;
    uint16_t left_idx = ht_node_alloc(&tree.memory);
    uint16_t right_idx = ht_node_alloc(&tree.memory);
    HT_ASSERT_NEQ(left_idx, HT_INVALID_IDX);
    HT_ASSERT_NEQ(right_idx, HT_INVALID_IDX);

    ht_node_t *r = &tree.memory.node_pool[root_idx];
    r->flags &= ~HT_NODE_IS_LEAF;
    r->left_child = left_idx;
    r->right_child = right_idx;
    if (r->stats_idx != HT_INVALID_IDX) {
        ht_stats_free(&tree.memory, r->stats_idx);
        r->stats_idx = HT_INVALID_IDX;
    }
    tree.n_decision_nodes = 1;
    tree.n_active_leaves = 2;

    /* Allocate stats for children */
    uint16_t l_stats = ht_stats_alloc(&tree.memory);
    uint16_t r_stats = ht_stats_alloc(&tree.memory);
    tree.memory.node_pool[left_idx].stats_idx = l_stats;
    tree.memory.node_pool[right_idx].stats_idx = r_stats;

    /* Allocate ADWIN for children */
    tree.node_drift[left_idx].adwin_idx = ht_adwin_pool_alloc(&ad_pool, NULL);
    tree.node_drift[right_idx].adwin_idx = ht_adwin_pool_alloc(&ad_pool, NULL);

    /* Now create an alternate subtree at root */
    uint16_t alt_root = ht_node_alloc(&tree.memory);
    HT_ASSERT_NEQ(alt_root, HT_INVALID_IDX);
    uint16_t alt_stats = ht_stats_alloc(&tree.memory);
    tree.memory.node_pool[alt_root].stats_idx = alt_stats;
    tree.node_drift[alt_root].adwin_idx = ht_adwin_pool_alloc(&ad_pool, NULL);
    tree.node_drift[root_idx].alt_tree_idx = alt_root;

    /* Total allocated before switch: 4 nodes, 3 stats (leaves only), 4 detectors */
    HT_ASSERT_EQ(tree.memory.nodes_allocated, 4U);
    HT_ASSERT_EQ(tree.memory.stats_allocated, 3U);
    HT_ASSERT_EQ(ad_pool.allocated_count, 4U);

    /* Perform SWITCH: alt_root becomes new root, old root subtree (root, left, right) freed */
    tree.root_idx = alt_root;
    tree.node_drift[root_idx].alt_tree_idx = HT_INVALID_IDX;
    HT_ASSERT_EQ(ht_tree_free_subtree(&tree, root_idx), HT_OK);

    /* After switch: exactly 1 node (alt_root), 1 stats, 1 detector remaining! */
    HT_ASSERT_EQ(tree.memory.nodes_allocated, 1U);
    HT_ASSERT_EQ(tree.memory.stats_allocated, 1U);
    HT_ASSERT_EQ(ad_pool.allocated_count, 1U);
    HT_ASSERT_EQ(tree.root_idx, alt_root);

    /* Clean up remaining tree */
    HT_ASSERT_EQ(ht_tree_free_subtree(&tree, tree.root_idx), HT_OK);
    HT_ASSERT_EQ(tree.memory.nodes_allocated, 0U);
    HT_ASSERT_EQ(tree.memory.stats_allocated, 0U);
    HT_ASSERT_EQ(ad_pool.allocated_count, 0U);
}

static void test_hat_alternate_tree_pruning_and_recycling(void) {
    ht_adwin_pool_t ad_pool;
    HT_ASSERT_EQ(ht_adwin_pool_init(&ad_pool), HT_OK);

    ht_config_params_t params;
    ht_default_hat_params(&params, 4, 2);

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init_hat(&tree, &params, &ad_pool), HT_OK);

    /* Attach alternate tree at root */
    uint16_t alt_root = ht_node_alloc(&tree.memory);
    uint16_t alt_stats = ht_stats_alloc(&tree.memory);
    tree.memory.node_pool[alt_root].stats_idx = alt_stats;
    tree.node_drift[alt_root].adwin_idx = ht_adwin_pool_alloc(&ad_pool, NULL);
    tree.node_drift[tree.root_idx].alt_tree_idx = alt_root;

    HT_ASSERT_EQ(tree.memory.nodes_allocated, 2U);
    HT_ASSERT_EQ(tree.memory.stats_allocated, 2U);
    HT_ASSERT_EQ(ad_pool.allocated_count, 2U);

    /* Prune alternate tree */
    HT_ASSERT_EQ(ht_tree_free_subtree(&tree, alt_root), HT_OK);
    tree.node_drift[tree.root_idx].alt_tree_idx = HT_INVALID_IDX;

    /* Memory returns strictly to baseline: 1 node, 1 stats, 1 detector */
    HT_ASSERT_EQ(tree.memory.nodes_allocated, 1U);
    HT_ASSERT_EQ(tree.memory.stats_allocated, 1U);
    HT_ASSERT_EQ(ad_pool.allocated_count, 1U);

    HT_ASSERT_EQ(ht_tree_free_subtree(&tree, tree.root_idx), HT_OK);
    HT_ASSERT_EQ(tree.memory.nodes_allocated, 0U);
    HT_ASSERT_EQ(ad_pool.allocated_count, 0U);
}

/* Master suite runner for test_drift */
void test_drift_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Detector ADWIN com Histograma Exponencial Fixo (ht_drift)");

    /* Nível 1: Unidade e Casos de Borda */
    HT_RUN_TEST(test_adwin_default_config_and_init);
    HT_RUN_TEST(test_adwin_single_and_incremental_stats);
    HT_RUN_TEST(test_adwin_bucket_compression_powers_of_two);
    HT_RUN_TEST(test_adwin_cut_epsilon_analytical);
    HT_RUN_TEST(test_adwin_edge_cases_and_null);

    /* Nível 1: HAT Algoritmos e Casos de Borda */
    HT_RUN_TEST(test_hat_unilateral_error_filter);
    HT_RUN_TEST(test_hat_switch_prune_bound_analytical);
    HT_RUN_TEST(test_hat_evaluate_alternate_switch_and_prune);
    HT_RUN_TEST(test_hat_weighted_voting_one_over_error_squared);

    /* Nível 2: Integração, Paridade River e Pool Estático */
    HT_RUN_TEST(test_adwin_drift_detection_step_change_river_parity);
    HT_RUN_TEST(test_adwin_drift_detection_multiple_shifts);
    HT_RUN_TEST(test_adwin_pool_alloc_free_lifecycle);
    HT_RUN_TEST(test_adwin_pool_concurrent_detectors);
    HT_RUN_TEST(test_adwin_pool_stress_50k_recycling);
    HT_RUN_TEST(test_adwin_pool_bytes_budget);

    /* Nível 2: HAT Substituição e Poda de Subárvores */
    HT_RUN_TEST(test_hat_subtree_replacement_and_iterative_recycling);
    HT_RUN_TEST(test_hat_alternate_tree_pruning_and_recycling);
}
