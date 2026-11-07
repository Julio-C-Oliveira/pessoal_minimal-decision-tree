/**
 * @file test_drift.c
 * @brief Unit, Integration, and Parity Tests for ADWIN Concept Drift Detector and Pool.
 *
 * Covers Level 1 (Unit: math, buckets, compression, cuts, edge cases),
 * and Level 2 (Integration: River parity, step change, multi-drift, and static pool lifecycle).
 */

#include "test_harness.h"
#include "ht_drift.h"

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

/* Master suite runner for test_drift */
void test_drift_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Detector ADWIN com Histograma Exponencial Fixo (ht_drift)");

    /* Nível 1: Unidade e Casos de Borda */
    HT_RUN_TEST(test_adwin_default_config_and_init);
    HT_RUN_TEST(test_adwin_single_and_incremental_stats);
    HT_RUN_TEST(test_adwin_bucket_compression_powers_of_two);
    HT_RUN_TEST(test_adwin_cut_epsilon_analytical);
    HT_RUN_TEST(test_adwin_edge_cases_and_null);

    /* Nível 2: Integração, Paridade River e Pool Estático */
    HT_RUN_TEST(test_adwin_drift_detection_step_change_river_parity);
    HT_RUN_TEST(test_adwin_drift_detection_multiple_shifts);
    HT_RUN_TEST(test_adwin_pool_alloc_free_lifecycle);
    HT_RUN_TEST(test_adwin_pool_concurrent_detectors);
    HT_RUN_TEST(test_adwin_pool_stress_50k_recycling);
    HT_RUN_TEST(test_adwin_pool_bytes_budget);
}
