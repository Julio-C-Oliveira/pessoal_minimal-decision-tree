/**
 * @file ht_prep.h
 * @brief Zero-malloc embedded streaming data preprocessing micro-pipelines (libht_prep).
 *
 * Implements deterministic online preprocessing components designed for edge microcontrollers:
 * 1. Hampel median + MAD filter (ht_filter_hampel_t) for industrial sensor spike rejection.
 * 2. Incremental online scalers (ht_scaler_t): Z-Score (Welford) and Min-Max.
 * 3. Space-Saving heavy-hitter categorical encoder (ht_encoder_freq_t) for high-cardinality streams.
 * 4. Ben-Haim & Tom-Tov non-parametric streaming quantizer and discretizer (ht_quantizer_t).
 * 5. Composable unified micro-pipeline (ht_prep_pipeline_t) attachable to ht_tree_t.
 *
 * Strictly adheres to C11 (-std=c11 -Wall -Wextra -Werror -pedantic) and zero runtime heap allocation.
 */

#ifndef HT_PREP_H
#define HT_PREP_H

#include "ht_config.h"
#include "ht_memory.h"

/* ========================================================================= */
/* --- 1. Online Scaler (StandardScaler & MinMaxScaler) -------------------- */
/* ========================================================================= */

typedef enum {
    HT_SCALER_NONE     = 0,
    HT_SCALER_STANDARD = 1, /**< z = (x - mean) / sqrt(var + eps) */
    HT_SCALER_MINMAX   = 2  /**< z = (x - min) / (max - min + eps) in [0, 1] */
} ht_scaler_type_t;

/**
 * @brief Online feature scaler managing Welford and min-max accumulators per feature.
 */
typedef struct {
    ht_gaussian_obs_t stats[HT_MAX_FEATURES]; /**< Reuses count, mean, m2, min_val, max_val */
    ht_scaler_type_t  mode[HT_MAX_FEATURES];
} ht_scaler_t;

void ht_scaler_init(ht_scaler_t *scaler);
void ht_scaler_set_mode(ht_scaler_t *scaler, uint8_t feat_idx, ht_scaler_type_t mode);
void ht_scaler_update_one(ht_scaler_t *scaler, uint8_t feat_idx, ht_float_t val, ht_float_t weight);
ht_float_t ht_scaler_transform_one(const ht_scaler_t *scaler, uint8_t feat_idx, ht_float_t val);
ht_float_t ht_scaler_learn_transform_one(ht_scaler_t *scaler, uint8_t feat_idx, ht_float_t val, ht_float_t weight);

/* ========================================================================= */
/* --- 2. Hampel Moving Median + MAD Outlier Filter ------------------------ */
/* ========================================================================= */

/**
 * @brief Sliding window median filter with Median Absolute Deviation (MAD) spike rejection.
 */
typedef struct {
    ht_float_t window[HT_MAX_FEATURES][HT_HAMPEL_WINDOW_SIZE];
    ht_float_t n_sigma;                       /**< Outlier threshold multiplier (default: 3.0) */
    uint32_t   outliers_replaced[HT_MAX_FEATURES];
    uint8_t    count[HT_MAX_FEATURES];
    uint8_t    head[HT_MAX_FEATURES];
    bool       enabled[HT_MAX_FEATURES];
} ht_filter_hampel_t;

void ht_hampel_init(ht_filter_hampel_t *f);
void ht_hampel_set_enabled(ht_filter_hampel_t *f, uint8_t feat_idx, bool enabled);
void ht_hampel_set_threshold(ht_filter_hampel_t *f, ht_float_t n_sigma);
ht_float_t ht_hampel_filter_one(ht_filter_hampel_t *f, uint8_t feat_idx, ht_float_t val);

/* ========================================================================= */
/* --- 3. Space-Saving High-Cardinality Categorical Encoder ---------------- */
/* ========================================================================= */

typedef enum {
    HT_ENC_NONE       = 0,
    HT_ENC_FREQ_RATIO = 1, /**< Outputs empirical relative frequency count/total in [0, 1] (.num) */
    HT_ENC_TOP_K_CAT  = 2  /**< Maps top-(K-1) frequent IDs to 0..K-2 and rest to K-1 (.cat) */
} ht_encoder_mode_t;

typedef struct {
    int32_t    key;
    ht_float_t count;
    uint8_t    occupied;
} ht_cat_slot_t;

typedef struct {
    ht_cat_slot_t     slots[HT_PREP_MAX_CAT_SLOTS];
    ht_float_t        total_weight;
    uint8_t           top_k;    /**< K categories (default: HT_MAX_CATEGORIES) */
    ht_encoder_mode_t mode;
} ht_feature_encoder_t;

typedef struct {
    ht_feature_encoder_t feat[HT_MAX_FEATURES];
} ht_encoder_freq_t;

void ht_encoder_init(ht_encoder_freq_t *enc);
void ht_encoder_set_mode(ht_encoder_freq_t *enc, uint8_t feat_idx, ht_encoder_mode_t mode, uint8_t top_k);
void ht_encoder_update_one(ht_encoder_freq_t *enc, uint8_t feat_idx, int32_t cat_val, ht_float_t weight);
ht_value_t ht_encoder_transform_one(const ht_encoder_freq_t *enc, uint8_t feat_idx, int32_t cat_val);
ht_value_t ht_encoder_learn_transform_one(ht_encoder_freq_t *enc, uint8_t feat_idx, int32_t cat_val, ht_float_t weight);

/* ========================================================================= */
/* --- 4. Ben-Haim & Tom-Tov Streaming Quantizer / Discretizer ------------- */
/* ========================================================================= */

typedef enum {
    HT_QUANT_NONE          = 0,
    HT_QUANT_CDF_UNIFORM   = 1, /**< Continuous rank/quantile mapping F(x) in [0, 1] (.num) */
    HT_QUANT_KBINS_ORDINAL = 2, /**< Discretizes into ordinal float 0.0 .. (n_bins - 1) (.num) */
    HT_QUANT_KBINS_CAT     = 3  /**< Discretizes into categorical index 0 .. (n_bins - 1) (.cat) */
} ht_quant_mode_t;

typedef struct {
    ht_float_t mean;
    ht_float_t weight;
} ht_centroid_t;

typedef struct {
    ht_centroid_t   centroids[HT_PREP_MAX_CENTROIDS];
    ht_float_t      min_val;
    ht_float_t      max_val;
    ht_float_t      total_weight;
    uint8_t         n_centroids;
    uint8_t         n_bins;     /**< Target bins K for KBINS modes (default: 5) */
    ht_quant_mode_t mode;
} ht_feature_quantizer_t;

typedef struct {
    ht_feature_quantizer_t feat[HT_MAX_FEATURES];
} ht_quantizer_t;

void ht_quantizer_init(ht_quantizer_t *q);
void ht_quantizer_set_mode(ht_quantizer_t *q, uint8_t feat_idx, ht_quant_mode_t mode, uint8_t n_bins);
void ht_quantizer_update_one(ht_quantizer_t *q, uint8_t feat_idx, ht_float_t val, ht_float_t weight);
ht_float_t ht_quantizer_cdf(const ht_feature_quantizer_t *fq, ht_float_t val);
ht_float_t ht_quantizer_quantile(const ht_feature_quantizer_t *fq, ht_float_t q_val);
ht_value_t ht_quantizer_transform_one(const ht_quantizer_t *q, uint8_t feat_idx, ht_float_t val);
ht_value_t ht_quantizer_learn_transform_one(ht_quantizer_t *q, uint8_t feat_idx, ht_float_t val, ht_float_t weight);

/* ========================================================================= */
/* --- 5. Composable Zero-Malloc Micro-Pipeline ---------------------------- */
/* ========================================================================= */

typedef struct ht_prep_pipeline {
    uint8_t            n_features;
    ht_feature_type_t  feature_types_in[HT_MAX_FEATURES];
    ht_feature_type_t  feature_types_out[HT_MAX_FEATURES];
    ht_filter_hampel_t hampel;
    ht_encoder_freq_t  encoder;
    ht_quantizer_t     quantizer;
    ht_scaler_t        scaler;
} ht_prep_pipeline_t;

void ht_prep_pipeline_init(ht_prep_pipeline_t *pipe, uint8_t n_features, const ht_feature_type_t feature_types[HT_MAX_FEATURES]);
void ht_prep_pipeline_update_types_out(ht_prep_pipeline_t *pipe);
void ht_prep_transform_one(const ht_prep_pipeline_t *pipe, const ht_value_t x_in[HT_MAX_FEATURES], ht_value_t x_out[HT_MAX_FEATURES]);
void ht_prep_learn_transform_one(ht_prep_pipeline_t *pipe, const ht_value_t x_in[HT_MAX_FEATURES], ht_value_t x_out[HT_MAX_FEATURES], ht_float_t weight);

#endif /* HT_PREP_H */
