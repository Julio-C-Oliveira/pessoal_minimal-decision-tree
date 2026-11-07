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

void test_e2e_flow_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 3: Fluxo Ponta a Ponta Acumulado (E2E)");
    HT_RUN_TEST(test_e2e_sample_vector_ingestion);
    HT_RUN_TEST(test_e2e_math_numerical_pipeline);
}
