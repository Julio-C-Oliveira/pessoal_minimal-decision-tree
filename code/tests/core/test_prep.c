/**
 * @file test_prep.c
 * @brief Unit and integration tests for streaming data preprocessing (libht_prep).
 */

#include "test_harness.h"
#include "ht_prep.h"
#include <string.h>

/* ========================================================================= */
/* --- 1. Scaler Unit Tests ------------------------------------------------ */
/* ========================================================================= */

static void test_scaler_welford_standard(void) {
    ht_scaler_t scaler;
    ht_scaler_init(&scaler);

    ht_scaler_set_mode(&scaler, 0, HT_SCALER_STANDARD);

    /* Feed 10, 20, 30, 40, 50 */
    ht_float_t vals[5] = {
        HT_CONST(10.0), HT_CONST(20.0), HT_CONST(30.0), HT_CONST(40.0), HT_CONST(50.0)
    };
    for (int i = 0; i < 5; ++i) {
        ht_scaler_update_one(&scaler, 0, vals[i], HT_CONST(1.0));
    }

    /* Mean should be 30.0, variance 250.0 (ddof=1) */
    HT_ASSERT_FLOAT_NEAR(scaler.stats[0].mean, HT_CONST(30.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(scaler.stats[0].count, HT_CONST(5.0), HT_TEST_TOL(1e-5));
    ht_float_t var = HT_DIV(scaler.stats[0].m2, HT_SUB(scaler.stats[0].count, HT_CONST(1.0)));
    HT_ASSERT_FLOAT_NEAR(var, HT_CONST(250.0), HT_TEST_TOL(1e-2));

    /* z(30.0) == 0.0 */
    ht_float_t z_mid = ht_scaler_transform_one(&scaler, 0, HT_CONST(30.0));
    HT_ASSERT_FLOAT_NEAR(z_mid, HT_CONST(0.0), HT_TEST_TOL(1e-4));

    /* z(10.0) == (10 - 30) / sqrt(250) ~= -1.264911 */
    ht_float_t z_low = ht_scaler_transform_one(&scaler, 0, HT_CONST(10.0));
    HT_ASSERT_FLOAT_NEAR(z_low, HT_CONST(-1.264911), HT_TEST_TOL(1e-3));

    /* z(50.0) == (50 - 30) / sqrt(250) ~= +1.264911 */
    ht_float_t z_high = ht_scaler_transform_one(&scaler, 0, HT_CONST(50.0));
    HT_ASSERT_FLOAT_NEAR(z_high, HT_CONST(1.264911), HT_TEST_TOL(1e-3));

    /* Edge case: NaN */
    ht_float_t nan_in = HT_NAN_VAL;
    ht_scaler_update_one(&scaler, 0, nan_in, HT_CONST(1.0));
    HT_ASSERT_FLOAT_NEAR(scaler.stats[0].count, HT_CONST(5.0), HT_TEST_TOL(1e-5));
    ht_float_t z_nan = ht_scaler_transform_one(&scaler, 0, nan_in);
    HT_ASSERT(HT_ISNAN(z_nan));
}

static void test_scaler_minmax(void) {
    ht_scaler_t scaler;
    ht_scaler_init(&scaler);

    ht_scaler_set_mode(&scaler, 1, HT_SCALER_MINMAX);

    ht_scaler_update_one(&scaler, 1, HT_CONST(10.0), HT_CONST(1.0));
    ht_scaler_update_one(&scaler, 1, HT_CONST(50.0), HT_CONST(1.0));
    ht_scaler_update_one(&scaler, 1, HT_CONST(5.0), HT_CONST(1.0));

    HT_ASSERT_FLOAT_NEAR(scaler.stats[1].min_val, HT_CONST(5.0), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(scaler.stats[1].max_val, HT_CONST(50.0), HT_TEST_TOL(1e-5));

    /* (5 - 5) / (50 - 5) == 0.0 */
    HT_ASSERT_FLOAT_NEAR(ht_scaler_transform_one(&scaler, 1, HT_CONST(5.0)), HT_CONST(0.0), HT_TEST_TOL(1e-5));
    /* (50 - 5) / (50 - 5) == 1.0 */
    HT_ASSERT_FLOAT_NEAR(ht_scaler_transform_one(&scaler, 1, HT_CONST(50.0)), HT_CONST(1.0), HT_TEST_TOL(1e-5));
    /* (27.5 - 5) / 45 == 0.5 */
    HT_ASSERT_FLOAT_NEAR(ht_scaler_transform_one(&scaler, 1, HT_CONST(27.5)), HT_CONST(0.5), HT_TEST_TOL(1e-5));
}

/* ========================================================================= */
/* --- 2. Hampel Filter Unit Tests ----------------------------------------- */
/* ========================================================================= */

static void test_hampel_spike_rejection(void) {
    ht_filter_hampel_t f;
    ht_hampel_init(&f);
    ht_hampel_set_enabled(&f, 0, true);
    ht_hampel_set_threshold(&f, HT_CONST(3.0));

    /* Feed steady signal around 10.0 */
    ht_float_t stream[7] = {
        HT_CONST(10.0), HT_CONST(10.1), HT_CONST(9.9),
        HT_CONST(10.0), HT_CONST(10.2), HT_CONST(9.8),
        HT_CONST(10.0)
    };
    for (int i = 0; i < 7; ++i) {
        ht_float_t out = ht_hampel_filter_one(&f, 0, stream[i]);
        HT_ASSERT_FLOAT_NEAR(out, stream[i], HT_TEST_TOL(0.01));
    }
    HT_ASSERT_EQ(f.outliers_replaced[0], 0U);

    /* Inject a massive spike of 500.0 */
    ht_float_t spike_out = ht_hampel_filter_one(&f, 0, HT_CONST(500.0));
    HT_ASSERT_EQ(f.outliers_replaced[0], 1U);
    /* Spike should have been replaced with the median (~10.0) */
    HT_ASSERT_FLOAT_NEAR(spike_out, HT_CONST(10.0), HT_TEST_TOL(0.2));

    /* Next normal sample should be passed without distortion */
    ht_float_t next_out = ht_hampel_filter_one(&f, 0, HT_CONST(10.05));
    HT_ASSERT_FLOAT_NEAR(next_out, HT_CONST(10.05), HT_TEST_TOL(0.01));
}

/* ========================================================================= */
/* --- 3. Space-Saving Encoder Unit Tests ----------------------------------- */
/* ========================================================================= */

static void test_encoder_space_saving(void) {
    ht_encoder_freq_t enc;
    ht_encoder_init(&enc);

    /* Feature 0: Top-K (K = 4) */
    ht_encoder_set_mode(&enc, 0, HT_ENC_TOP_K_CAT, 4);

    /* Feed 3 heavy hitters (IDs 100, 200, 300) and 20 rare categories */
    for (int i = 0; i < 50; ++i) {
        ht_encoder_update_one(&enc, 0, 100, HT_CONST(1.0));
    }
    for (int i = 0; i < 30; ++i) {
        ht_encoder_update_one(&enc, 0, 200, HT_CONST(1.0));
    }
    for (int i = 0; i < 20; ++i) {
        ht_encoder_update_one(&enc, 0, 300, HT_CONST(1.0));
    }
    for (int i = 1; i <= 20; ++i) {
        ht_encoder_update_one(&enc, 0, 1000 + i, HT_CONST(1.0));
    }

    /* Top-3 categories should be mapped to 0, 1, 2 */
    ht_value_t out_100 = ht_encoder_transform_one(&enc, 0, 100);
    ht_value_t out_200 = ht_encoder_transform_one(&enc, 0, 200);
    ht_value_t out_300 = ht_encoder_transform_one(&enc, 0, 300);
    HT_ASSERT_EQ(out_100.cat, 0);
    HT_ASSERT_EQ(out_200.cat, 1);
    HT_ASSERT_EQ(out_300.cat, 2);

    /* Rare categories should map to bucket K-1 = 3 ("Others") */
    ht_value_t out_rare = ht_encoder_transform_one(&enc, 0, 1005);
    HT_ASSERT_EQ(out_rare.cat, 3);
    ht_value_t out_unseen = ht_encoder_transform_one(&enc, 0, 9999);
    HT_ASSERT_EQ(out_unseen.cat, 3);

    /* Feature 1: Frequency Ratio */
    ht_encoder_set_mode(&enc, 1, HT_ENC_FREQ_RATIO, 4);
    for (int i = 0; i < 75; ++i) {
        ht_encoder_update_one(&enc, 1, 42, HT_CONST(1.0));
    }
    for (int i = 0; i < 25; ++i) {
        ht_encoder_update_one(&enc, 1, 84, HT_CONST(1.0));
    }
    ht_value_t ratio_42 = ht_encoder_transform_one(&enc, 1, 42);
    HT_ASSERT_FLOAT_NEAR(ratio_42.num, HT_CONST(0.75), HT_TEST_TOL(1e-4));
    ht_value_t ratio_84 = ht_encoder_transform_one(&enc, 1, 84);
    HT_ASSERT_FLOAT_NEAR(ratio_84.num, HT_CONST(0.25), HT_TEST_TOL(1e-4));
}

/* ========================================================================= */
/* --- 4. Ben-Haim & Tom-Tov Quantizer Unit Tests -------------------------- */
/* ========================================================================= */

static void test_quantizer_ben_haim_cdf_and_kbins(void) {
    ht_quantizer_t q;
    ht_quantizer_init(&q);

    /* Uniform distribution in [0, 100] */
    ht_quantizer_set_mode(&q, 0, HT_QUANT_CDF_UNIFORM, 5);
    for (int i = 0; i <= 100; ++i) {
        ht_quantizer_update_one(&q, 0, HT_FROM_INT(i), HT_CONST(1.0));
    }

    /* Check CDF at 0, 50, 100 */
    ht_float_t cdf_0 = ht_quantizer_cdf(&q.feat[0], HT_CONST(0.0));
    ht_float_t cdf_50 = ht_quantizer_cdf(&q.feat[0], HT_CONST(50.0));
    ht_float_t cdf_100 = ht_quantizer_cdf(&q.feat[0], HT_CONST(100.0));

    HT_ASSERT_FLOAT_NEAR(cdf_0, HT_CONST(0.0), HT_TEST_TOL(0.05));
    HT_ASSERT_FLOAT_NEAR(cdf_50, HT_CONST(0.5), HT_TEST_TOL(0.05));
    HT_ASSERT_FLOAT_NEAR(cdf_100, HT_CONST(1.0), HT_TEST_TOL(0.05));

    /* Invert quantiles */
    ht_float_t q25 = ht_quantizer_quantile(&q.feat[0], HT_CONST(0.25));
    ht_float_t q50 = ht_quantizer_quantile(&q.feat[0], HT_CONST(0.50));
    ht_float_t q75 = ht_quantizer_quantile(&q.feat[0], HT_CONST(0.75));

    HT_ASSERT_FLOAT_NEAR(q25, HT_CONST(25.0), HT_TEST_TOL(3.0));
    HT_ASSERT_FLOAT_NEAR(q50, HT_CONST(50.0), HT_TEST_TOL(3.0));
    HT_ASSERT_FLOAT_NEAR(q75, HT_CONST(75.0), HT_TEST_TOL(3.0));

    /* KBins Categorical */
    ht_quantizer_set_mode(&q, 1, HT_QUANT_KBINS_CAT, 5);
    for (int i = 0; i <= 100; ++i) {
        ht_quantizer_update_one(&q, 1, HT_FROM_INT(i), HT_CONST(1.0));
    }

    ht_value_t b0 = ht_quantizer_transform_one(&q, 1, HT_CONST(5.0));
    ht_value_t b2 = ht_quantizer_transform_one(&q, 1, HT_CONST(50.0));
    ht_value_t b4 = ht_quantizer_transform_one(&q, 1, HT_CONST(95.0));

    HT_ASSERT_EQ(b0.cat, 0);
    HT_ASSERT_EQ(b2.cat, 2);
    HT_ASSERT_EQ(b4.cat, 4);
}

/* ========================================================================= */
/* --- 5. Integrated Pipeline Tests ---------------------------------------- */
/* ========================================================================= */

static void test_prep_pipeline_full_flow(void) {
    ht_prep_pipeline_t pipe;
    ht_feature_type_t types[HT_MAX_FEATURES];
    memset(types, 0, sizeof(types));
    types[0] = HT_FEAT_NUMERIC;
    types[1] = HT_FEAT_NUMERIC;
    types[2] = HT_FEAT_CATEGORICAL;
    ht_prep_pipeline_init(&pipe, 3, types);

    /* Feature 0: Hampel + StandardScaler */
    ht_hampel_set_enabled(&pipe.hampel, 0, true);
    ht_scaler_set_mode(&pipe.scaler, 0, HT_SCALER_STANDARD);

    /* Feature 1: Quantizer KBins (Categorical output) */
    ht_quantizer_set_mode(&pipe.quantizer, 1, HT_QUANT_KBINS_CAT, 5);

    /* Feature 2: Space-Saving Top-K */
    ht_encoder_set_mode(&pipe.encoder, 2, HT_ENC_TOP_K_CAT, 3);

    ht_prep_pipeline_update_types_out(&pipe);
    HT_ASSERT_EQ(pipe.feature_types_out[0], HT_FEAT_NUMERIC);
    HT_ASSERT_EQ(pipe.feature_types_out[1], HT_FEAT_CATEGORICAL);
    HT_ASSERT_EQ(pipe.feature_types_out[2], HT_FEAT_CATEGORICAL);

    /* Ingest streaming data */
    for (int i = 0; i < 100; ++i) {
        ht_value_t in[HT_MAX_FEATURES];
        ht_value_t out[HT_MAX_FEATURES];
        memset(in, 0, sizeof(in));
        memset(out, 0, sizeof(out));

        in[0].num = HT_FROM_INT(i % 10);
        in[1].num = HT_FROM_INT(i);
        in[2].cat = (i % 2 == 0) ? 100 : ((i % 3 == 0) ? 200 : (1000 + i));

        ht_prep_learn_transform_one(&pipe, in, out, HT_CONST(1.0));

        HT_ASSERT(!HT_ISNAN(out[0].num));
        HT_ASSERT(out[1].cat >= 0 && out[1].cat < 5);
        HT_ASSERT(out[2].cat >= 0 && out[2].cat < 3);
    }

    /* Transform a test sample with a massive spike on feature 0 */
    ht_value_t test_in[HT_MAX_FEATURES];
    ht_value_t test_out[HT_MAX_FEATURES];
    memset(test_in, 0, sizeof(test_in));
    memset(test_out, 0, sizeof(test_out));
    test_in[0].num = HT_CONST(9999.0); /* Outlier! */
    test_in[1].num = HT_CONST(50.0);
    test_in[2].cat = 100;

    ht_prep_transform_one(&pipe, test_in, test_out);

    /* Feature 0 should be cleaned by Hampel and standardized around 0.0 */
    HT_ASSERT(HT_FABS(test_out[0].num) < HT_CONST(2.5));
    HT_ASSERT_EQ(test_out[1].cat, 2);
    HT_ASSERT_EQ(test_out[2].cat, 0);
}

static void test_prep_edge_cases_nan_high_scale_and_quantizer_boundaries(void) {
    /* 1. Scaler edge cases: NULL, invalid feature, NaN, constant variance, high-scale Q16.16 */
    ht_scaler_t sc;
    ht_scaler_init(&sc);
    ht_scaler_set_mode(&sc, 0, HT_SCALER_STANDARD);
    ht_scaler_set_mode(&sc, 1, HT_SCALER_MINMAX);

    HT_ASSERT(HT_ISNAN(ht_scaler_transform_one(&sc, 0, HT_NAN_VAL)));
    HT_ASSERT_FLOAT_EQ(ht_scaler_transform_one(&sc, 0, HT_CONST(5.0)), HT_CONST(0.0)); /* n < 2 */
    HT_ASSERT_FLOAT_EQ(ht_scaler_transform_one(&sc, 1, HT_CONST(5.0)), HT_CONST(0.0)); /* min == max */
    HT_ASSERT_FLOAT_EQ(ht_scaler_transform_one(&sc, 99, HT_CONST(7.0)), HT_CONST(7.0)); /* out-of-range */
    ht_scaler_update_one(&sc, 0, HT_NAN_VAL, HT_CONST(1.0));

    /* Constant values: variance == 0 */
    for (int i = 0; i < 10; ++i) {
        ht_scaler_update_one(&sc, 0, HT_CONST(42.0), HT_CONST(1.0));
        ht_scaler_update_one(&sc, 1, HT_CONST(42.0), HT_CONST(1.0));
    }
    HT_ASSERT_FLOAT_EQ(ht_scaler_transform_one(&sc, 0, HT_CONST(42.0)), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_scaler_transform_one(&sc, 1, HT_CONST(42.0)), HT_CONST(0.0));

    /* High-amplitude + high-weight updates triggering Q16.16 rescaling in scaler */
    for (int i = 0; i < 200; ++i) {
        ht_float_t v = ((i & 1) == 0) ? HT_CONST(100.0) : HT_CONST(900.0);
        ht_scaler_update_one(&sc, 0, v, HT_CONST(100.0));
    }
    HT_ASSERT(HT_ISFINITE(ht_scaler_transform_one(&sc, 0, HT_CONST(500.0))));

    /* 2. Hampel edge cases: NULL, NaN, out-of-range feature */
    ht_filter_hampel_t hf;
    ht_hampel_init(&hf);
    ht_hampel_set_enabled(&hf, 0, true);
    HT_ASSERT(HT_ISNAN(ht_hampel_filter_one(&hf, 0, HT_NAN_VAL)));
    HT_ASSERT_FLOAT_EQ(ht_hampel_filter_one(&hf, 99, HT_CONST(3.0)), HT_CONST(3.0));

    /* 3. Space-Saving encoder: K > HT_MAX_CATEGORIES clamping, rescaling & unseen category */
    ht_encoder_freq_t enc;
    ht_encoder_init(&enc);
    ht_encoder_set_mode(&enc, 0, HT_ENC_TOP_K_CAT, 100U); /* clamps to HT_MAX_CATEGORIES */
    ht_encoder_set_mode(&enc, 1, HT_ENC_FREQ_RATIO, 3U);
    ht_encoder_update_one(&enc, 0, -1, HT_CONST(1.0)); /* missing cat ignored */
    for (int i = 0; i < 200; ++i) {
        ht_encoder_update_one(&enc, 0, 10, HT_CONST(100.0));
        ht_encoder_update_one(&enc, 1, 20, HT_CONST(100.0));
    }
    ht_value_t eo0 = ht_encoder_transform_one(&enc, 0, 10);
    ht_value_t eo_unseen = ht_encoder_transform_one(&enc, 0, 99999);
    ht_value_t eo_freq = ht_encoder_transform_one(&enc, 1, 20);
    ht_value_t eo_miss = ht_encoder_transform_one(&enc, 0, -1);
    HT_ASSERT_EQ(eo0.cat, 0);
    HT_ASSERT(eo_unseen.cat >= 0);
    HT_ASSERT(eo_freq.num > HT_CONST(0.5));
    HT_ASSERT_EQ(eo_miss.cat, -1);

    /* 4. Quantizer boundary branches: empty, 1 centroid, below-min, above-max, exact match, left insert, NaN */
    ht_quantizer_t q;
    ht_quantizer_init(&q);
    ht_quantizer_set_mode(&q, 0, HT_QUANT_CDF_UNIFORM, 4);
    ht_quantizer_set_mode(&q, 1, HT_QUANT_KBINS_CAT, 4);
    ht_quantizer_set_mode(&q, 2, HT_QUANT_KBINS_ORDINAL, 4);

    /* Empty quantizer */
    HT_ASSERT_FLOAT_EQ(ht_quantizer_transform_one(&q, 0, HT_CONST(5.0)).num, HT_CONST(0.5));
    HT_ASSERT_EQ(ht_quantizer_transform_one(&q, 1, HT_CONST(5.0)).cat, 2);
    HT_ASSERT(HT_ISNAN(ht_quantizer_transform_one(&q, 0, HT_NAN_VAL).num));
    HT_ASSERT_EQ(ht_quantizer_transform_one(&q, 1, HT_NAN_VAL).cat, -1);
    HT_ASSERT(HT_ISNAN(ht_quantizer_transform_one(&q, 2, HT_NAN_VAL).num));

    /* Single centroid */
    ht_quantizer_update_one(&q, 0, HT_CONST(10.0), HT_CONST(1.0));
    ht_quantizer_update_one(&q, 0, HT_CONST(10.0), HT_CONST(2.0)); /* exact match */
    HT_ASSERT_FLOAT_EQ(ht_quantizer_transform_one(&q, 0, HT_CONST(5.0)).num, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_quantizer_transform_one(&q, 0, HT_CONST(10.0)).num, HT_CONST(0.5));
    HT_ASSERT_FLOAT_EQ(ht_quantizer_transform_one(&q, 0, HT_CONST(20.0)).num, HT_CONST(1.0));

    /* Left insert + right insert + below-min + above-max + ordinal bins + quantile */
    ht_quantizer_update_one(&q, 0, HT_CONST(2.0), HT_CONST(1.0));  /* left insert */
    ht_quantizer_update_one(&q, 0, HT_CONST(20.0), HT_CONST(1.0)); /* right insert */
    ht_quantizer_update_one(&q, 0, HT_NAN_VAL, HT_CONST(1.0));     /* NaN ignored */
    HT_ASSERT_FLOAT_EQ(ht_quantizer_transform_one(&q, 0, HT_CONST(-10.0)).num, HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_quantizer_transform_one(&q, 0, HT_CONST(100.0)).num, HT_CONST(1.0));
    ht_float_t mid_cdf = ht_quantizer_transform_one(&q, 0, HT_CONST(6.0)).num;
    HT_ASSERT(mid_cdf > HT_CONST(0.0) && mid_cdf < HT_CONST(1.0));

    ht_float_t q_med = ht_quantizer_quantile(&q.feat[0], HT_CONST(0.5));
    HT_ASSERT(q_med >= HT_CONST(2.0) && q_med <= HT_CONST(20.0));

    for (int i = 0; i < 40; ++i) {
        ht_quantizer_update_one(&q, 2, HT_FROM_INT(i), HT_CONST(1.0));
    }
    ht_float_t ord_lo = ht_quantizer_transform_one(&q, 2, HT_CONST(0.0)).num;
    ht_float_t ord_hi = ht_quantizer_transform_one(&q, 2, HT_CONST(39.0)).num;
    HT_ASSERT(ord_lo >= HT_CONST(0.0) && ord_hi <= HT_CONST(3.0));
    HT_ASSERT(ord_hi > ord_lo);
}

void test_prep_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Micro-Pipelines de Pre-Processamento (ht_prep)");
    HT_RUN_TEST(test_scaler_welford_standard);
    HT_RUN_TEST(test_scaler_minmax);
    HT_RUN_TEST(test_hampel_spike_rejection);
    HT_RUN_TEST(test_encoder_space_saving);
    HT_RUN_TEST(test_quantizer_ben_haim_cdf_and_kbins);
    HT_RUN_TEST(test_prep_pipeline_full_flow);
    HT_RUN_TEST(test_prep_edge_cases_nan_high_scale_and_quantizer_boundaries);
}
