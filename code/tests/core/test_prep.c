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
    ht_float_t vals[5] = { (ht_float_t)10.0, (ht_float_t)20.0, (ht_float_t)30.0, (ht_float_t)40.0, (ht_float_t)50.0 };
    for (int i = 0; i < 5; ++i) {
        ht_scaler_update_one(&scaler, 0, vals[i], (ht_float_t)1.0);
    }

    /* Mean should be 30.0, variance 250.0 (ddof=1) */
    HT_ASSERT_FLOAT_NEAR(scaler.stats[0].mean, (ht_float_t)30.0, (ht_float_t)1e-5);
    HT_ASSERT_FLOAT_NEAR(scaler.stats[0].count, (ht_float_t)5.0, (ht_float_t)1e-5);
    ht_float_t var = scaler.stats[0].m2 / (scaler.stats[0].count - (ht_float_t)1.0);
    HT_ASSERT_FLOAT_NEAR(var, (ht_float_t)250.0, (ht_float_t)1e-4);

    /* z(30.0) == 0.0 */
    ht_float_t z_mid = ht_scaler_transform_one(&scaler, 0, (ht_float_t)30.0);
    HT_ASSERT_FLOAT_NEAR(z_mid, (ht_float_t)0.0, (ht_float_t)1e-4);

    /* z(10.0) == (10 - 30) / sqrt(250) ≈ -1.264911 */
    ht_float_t z_low = ht_scaler_transform_one(&scaler, 0, (ht_float_t)10.0);
    HT_ASSERT_FLOAT_NEAR(z_low, (ht_float_t)-1.264911, (ht_float_t)1e-3);

    /* z(50.0) == (50 - 30) / sqrt(250) ≈ +1.264911 */
    ht_float_t z_high = ht_scaler_transform_one(&scaler, 0, (ht_float_t)50.0);
    HT_ASSERT_FLOAT_NEAR(z_high, (ht_float_t)1.264911, (ht_float_t)1e-3);

    /* Edge case: NaN */
    ht_float_t nan_in = (ht_float_t)0.0 / (ht_float_t)0.0;
    ht_scaler_update_one(&scaler, 0, nan_in, (ht_float_t)1.0);
    HT_ASSERT_FLOAT_NEAR(scaler.stats[0].count, (ht_float_t)5.0, (ht_float_t)1e-5); /* Count unchanged */
    ht_float_t z_nan = ht_scaler_transform_one(&scaler, 0, nan_in);
    HT_ASSERT(HT_ISNAN(z_nan));
}

static void test_scaler_minmax(void) {
    ht_scaler_t scaler;
    ht_scaler_init(&scaler);

    ht_scaler_set_mode(&scaler, 1, HT_SCALER_MINMAX);

    ht_scaler_update_one(&scaler, 1, (ht_float_t)10.0, (ht_float_t)1.0);
    ht_scaler_update_one(&scaler, 1, (ht_float_t)50.0, (ht_float_t)1.0);
    ht_scaler_update_one(&scaler, 1, (ht_float_t)5.0, (ht_float_t)1.0);

    HT_ASSERT_FLOAT_NEAR(scaler.stats[1].min_val, (ht_float_t)5.0, (ht_float_t)1e-5);
    HT_ASSERT_FLOAT_NEAR(scaler.stats[1].max_val, (ht_float_t)50.0, (ht_float_t)1e-5);

    /* (5 - 5) / (50 - 5) == 0.0 */
    HT_ASSERT_FLOAT_NEAR(ht_scaler_transform_one(&scaler, 1, (ht_float_t)5.0), (ht_float_t)0.0, (ht_float_t)1e-5);
    /* (50 - 5) / (50 - 5) == 1.0 */
    HT_ASSERT_FLOAT_NEAR(ht_scaler_transform_one(&scaler, 1, (ht_float_t)50.0), (ht_float_t)1.0, (ht_float_t)1e-5);
    /* (27.5 - 5) / 45 == 0.5 */
    HT_ASSERT_FLOAT_NEAR(ht_scaler_transform_one(&scaler, 1, (ht_float_t)27.5), (ht_float_t)0.5, (ht_float_t)1e-5);
}

/* ========================================================================= */
/* --- 2. Hampel Filter Unit Tests ----------------------------------------- */
/* ========================================================================= */

static void test_hampel_spike_rejection(void) {
    ht_filter_hampel_t f;
    ht_hampel_init(&f);
    ht_hampel_set_enabled(&f, 0, true);
    ht_hampel_set_threshold(&f, (ht_float_t)3.0);

    /* Feed steady signal around 10.0 */
    ht_float_t stream[7] = {
        (ht_float_t)10.0, (ht_float_t)10.1, (ht_float_t)9.9,
        (ht_float_t)10.0, (ht_float_t)10.2, (ht_float_t)9.8,
        (ht_float_t)10.0
    };
    for (int i = 0; i < 7; ++i) {
        ht_float_t out = ht_hampel_filter_one(&f, 0, stream[i]);
        HT_ASSERT_FLOAT_NEAR(out, stream[i], (ht_float_t)0.01);
    }
    HT_ASSERT_EQ(f.outliers_replaced[0], 0U);

    /* Inject a massive spike of 500.0 */
    ht_float_t spike_out = ht_hampel_filter_one(&f, 0, (ht_float_t)500.0);
    HT_ASSERT_EQ(f.outliers_replaced[0], 1U);
    /* Spike should have been replaced with the median (~10.0) */
    HT_ASSERT_FLOAT_NEAR(spike_out, (ht_float_t)10.0, (ht_float_t)0.2);

    /* Next normal sample should be passed without distortion */
    ht_float_t next_out = ht_hampel_filter_one(&f, 0, (ht_float_t)10.05);
    HT_ASSERT_FLOAT_NEAR(next_out, (ht_float_t)10.05, (ht_float_t)0.01);
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
        ht_encoder_update_one(&enc, 0, 100, (ht_float_t)1.0);
    }
    for (int i = 0; i < 30; ++i) {
        ht_encoder_update_one(&enc, 0, 200, (ht_float_t)1.0);
    }
    for (int i = 0; i < 20; ++i) {
        ht_encoder_update_one(&enc, 0, 300, (ht_float_t)1.0);
    }
    for (int i = 1; i <= 20; ++i) {
        ht_encoder_update_one(&enc, 0, 1000 + i, (ht_float_t)1.0);
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
        ht_encoder_update_one(&enc, 1, 42, (ht_float_t)1.0);
    }
    for (int i = 0; i < 25; ++i) {
        ht_encoder_update_one(&enc, 1, 84, (ht_float_t)1.0);
    }
    ht_value_t ratio_42 = ht_encoder_transform_one(&enc, 1, 42);
    HT_ASSERT_FLOAT_NEAR(ratio_42.num, (ht_float_t)0.75, (ht_float_t)1e-4);
    ht_value_t ratio_84 = ht_encoder_transform_one(&enc, 1, 84);
    HT_ASSERT_FLOAT_NEAR(ratio_84.num, (ht_float_t)0.25, (ht_float_t)1e-4);
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
        ht_quantizer_update_one(&q, 0, (ht_float_t)i, (ht_float_t)1.0);
    }

    /* Check CDF at 0, 50, 100 */
    ht_float_t cdf_0 = ht_quantizer_cdf(&q.feat[0], (ht_float_t)0.0);
    ht_float_t cdf_50 = ht_quantizer_cdf(&q.feat[0], (ht_float_t)50.0);
    ht_float_t cdf_100 = ht_quantizer_cdf(&q.feat[0], (ht_float_t)100.0);

    HT_ASSERT_FLOAT_NEAR(cdf_0, (ht_float_t)0.0, (ht_float_t)0.05);
    HT_ASSERT_FLOAT_NEAR(cdf_50, (ht_float_t)0.5, (ht_float_t)0.05);
    HT_ASSERT_FLOAT_NEAR(cdf_100, (ht_float_t)1.0, (ht_float_t)0.05);

    /* Invert quantiles */
    ht_float_t q25 = ht_quantizer_quantile(&q.feat[0], (ht_float_t)0.25);
    ht_float_t q50 = ht_quantizer_quantile(&q.feat[0], (ht_float_t)0.50);
    ht_float_t q75 = ht_quantizer_quantile(&q.feat[0], (ht_float_t)0.75);

    HT_ASSERT_FLOAT_NEAR(q25, (ht_float_t)25.0, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_NEAR(q50, (ht_float_t)50.0, (ht_float_t)3.0);
    HT_ASSERT_FLOAT_NEAR(q75, (ht_float_t)75.0, (ht_float_t)3.0);

    /* KBins Categorical */
    ht_quantizer_set_mode(&q, 1, HT_QUANT_KBINS_CAT, 5);
    for (int i = 0; i <= 100; ++i) {
        ht_quantizer_update_one(&q, 1, (ht_float_t)i, (ht_float_t)1.0);
    }

    ht_value_t b0 = ht_quantizer_transform_one(&q, 1, (ht_float_t)5.0);
    ht_value_t b2 = ht_quantizer_transform_one(&q, 1, (ht_float_t)50.0);
    ht_value_t b4 = ht_quantizer_transform_one(&q, 1, (ht_float_t)95.0);

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

        in[0].num = (ht_float_t)(i % 10);
        in[1].num = (ht_float_t)i;
        in[2].cat = (i % 2 == 0) ? 100 : ((i % 3 == 0) ? 200 : (1000 + i));

        ht_prep_learn_transform_one(&pipe, in, out, (ht_float_t)1.0);

        HT_ASSERT(!HT_ISNAN(out[0].num));
        HT_ASSERT(out[1].cat >= 0 && out[1].cat < 5);
        HT_ASSERT(out[2].cat >= 0 && out[2].cat < 3);
    }

    /* Transform a test sample with a massive spike on feature 0 */
    ht_value_t test_in[HT_MAX_FEATURES];
    ht_value_t test_out[HT_MAX_FEATURES];
    memset(test_in, 0, sizeof(test_in));
    memset(test_out, 0, sizeof(test_out));
    test_in[0].num = (ht_float_t)9999.0; /* Outlier! */
    test_in[1].num = (ht_float_t)50.0;
    test_in[2].cat = 100;

    ht_prep_transform_one(&pipe, test_in, test_out);

    /* Feature 0 should be cleaned by Hampel and standardized around 0.0 */
    HT_ASSERT(HT_FABS(test_out[0].num) < (ht_float_t)2.5);
    HT_ASSERT_EQ(test_out[1].cat, 2);
    HT_ASSERT_EQ(test_out[2].cat, 0);
}

void test_prep_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Micro-Pipelines de Pre-Processamento (ht_prep)");
    HT_RUN_TEST(test_scaler_welford_standard);
    HT_RUN_TEST(test_scaler_minmax);
    HT_RUN_TEST(test_hampel_spike_rejection);
    HT_RUN_TEST(test_encoder_space_saving);
    HT_RUN_TEST(test_quantizer_ben_haim_cdf_and_kbins);
    HT_RUN_TEST(test_prep_pipeline_full_flow);
}
