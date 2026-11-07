/**
 * @file ht_drift.h
 * @brief Fixed-size Exponential Histogram ADWIN Concept Drift Detector and Pool.
 *
 * Implements the ADWIN2 (Adaptive Windowing) algorithm using static exponential
 * histograms with O(log W) memory and zero dynamic allocation.
 *
 * Designed for embedded real-time systems (<= 1 MB RAM) and full parity
 * with River's drift.ADWIN / AdaptiveWindowing engine.
 */

#ifndef HT_DRIFT_H
#define HT_DRIFT_H

#include "ht_config.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief One level of exponential histogram buckets.
 *
 * Each level stores buckets of fixed capacity 2^level.
 * A level can hold up to (HT_ADWIN_MAX_BUCKETS + 1) buckets before compression.
 */
typedef struct {
    ht_float_t sum[HT_ADWIN_MAX_BUCKETS + 1];      /* Sum of elements in each bucket */
    ht_float_t variance[HT_ADWIN_MAX_BUCKETS + 1]; /* Within-bucket variance (M2) */
    uint8_t    count;                              /* Active buckets in this level (0..6) */
#ifndef HT_USE_DOUBLE
    uint8_t    pad[3];                             /* 32-bit alignment padding */
#else
    uint8_t    pad[7];                             /* 64-bit alignment padding */
#endif
} ht_adwin_row_t;

#ifndef HT_USE_DOUBLE
_Static_assert(sizeof(ht_adwin_row_t) == 52, "ht_adwin_row_t must be exactly 52 bytes in float32");
#else
_Static_assert(sizeof(ht_adwin_row_t) == 104, "ht_adwin_row_t must be exactly 104 bytes in float64");
#endif

/**
 * @brief Configuration parameters for an ADWIN detector.
 */
typedef struct {
    ht_float_t delta;               /* Confidence / significance parameter (default: 0.002) */
    uint16_t   clock;               /* Evaluation period in steps (default: 32) */
    uint16_t   min_window_length;   /* Min subwindow length for drift test (default: 5) */
    uint16_t   grace_period;        /* Min window size before checking drift (default: 10) */
    uint8_t    max_buckets;         /* Max buckets per level before merge (default: 5) */
    uint8_t    pad;
} ht_adwin_config_t;

/**
 * @brief Standalone static ADWIN detector instance.
 */
typedef struct {
    ht_adwin_row_t rows[HT_ADWIN_LEVELS]; /* Logarithmic exponential histogram rows */
    ht_float_t     total_sum;            /* Sum of all elements in current window */
    ht_float_t     total_variance;       /* Total M2 variance in current window */
    ht_float_t     delta;                /* Active significance parameter */
    uint32_t       width;                /* Current window size (total elements) */
    uint32_t       mint_time;            /* Total updates observed since init/reset */
    uint32_t       n_detections;         /* Total number of detected drifts */
    uint16_t       clock;                /* Evaluation period */
    uint16_t       min_window_length;    /* Minimum subwindow size */
    uint16_t       grace_period;         /* Grace period */
    uint8_t        max_buckets;          /* Maximum buckets per level */
    uint8_t        last_bucket_row;      /* Highest active row index */
    bool           drift_detected;       /* Drift flag set by most recent update */
    uint8_t        pad[3];               /* Explicit alignment padding */
} ht_adwin_t;

#ifndef HT_USE_DOUBLE
_Static_assert(sizeof(ht_adwin_t) == 764, "ht_adwin_t must be exactly 764 bytes in float32");
#else
_Static_assert(sizeof(ht_adwin_t) == 1504, "ht_adwin_t must be exactly 1504 bytes in float64");
#endif

/**
 * @brief Deterministic pool allocator for multiple ADWIN detector instances.
 */
typedef struct {
    ht_adwin_t pool[HT_MAX_ADWIN_DETECTORS];
    uint16_t   free_indices[HT_MAX_ADWIN_DETECTORS];
    uint32_t   in_use_mask[(HT_MAX_ADWIN_DETECTORS + 31U) / 32U];
    uint16_t   free_top;
    uint16_t   allocated_count;
} ht_adwin_pool_t;

_Static_assert(sizeof(ht_adwin_pool_t) <= (256 * 1024), "ht_adwin_pool_t must stay within budget");

/* --- Public ADWIN Detector API --- */

/**
 * @brief Populates default configuration parameters for ADWIN (matching River).
 *
 * @param cfg Pointer to config structure to initialize.
 */
void ht_adwin_default_config(ht_adwin_config_t *cfg);

/**
 * @brief Initializes an ADWIN instance with given configuration.
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @param cfg   Pointer to configuration (or NULL to use defaults).
 * @return HT_OK on success, or HT_ERR_INVALID_ARG if adwin is NULL.
 */
ht_status_t ht_adwin_init(ht_adwin_t *adwin, const ht_adwin_config_t *cfg);

/**
 * @brief Resets the window state of an ADWIN detector, preserving configuration.
 *
 * @param adwin Pointer to ADWIN detector instance.
 */
void ht_adwin_reset(ht_adwin_t *adwin);

/**
 * @brief Updates the ADWIN detector with a new stream observation.
 *
 * Inserts the value into the exponential histogram, compresses buckets as needed,
 * and periodically evaluates cut hypotheses. If concept drift is detected, the window
 * is shrunk and true is returned.
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @param value Scalar stream value to observe.
 * @return true if concept drift occurred at this step, false otherwise.
 */
bool ht_adwin_update(ht_adwin_t *adwin, ht_float_t value);

/**
 * @brief Returns the estimated mean of values within the current window.
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @return Estimated mean (or 0.0 if window is empty).
 */
ht_float_t ht_adwin_get_estimation(const ht_adwin_t *adwin);

/**
 * @brief Returns the cumulative M2 variance (sum of squared deviations) in window.
 *
 * Matches River's get_variance().
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @return Total M2 variance.
 */
ht_float_t ht_adwin_get_variance(const ht_adwin_t *adwin);

/**
 * @brief Returns the sample variance (M2 / width) within the current window.
 *
 * Matches River's variance_in_window property.
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @return Sample variance \sigma^2.
 */
ht_float_t ht_adwin_get_variance_in_window(const ht_adwin_t *adwin);

/**
 * @brief Returns the current window width (total observed elements stored).
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @return Window size in elements.
 */
uint32_t ht_adwin_get_width(const ht_adwin_t *adwin);

/**
 * @brief Returns the total sum of elements currently stored in window.
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @return Window sum.
 */
ht_float_t ht_adwin_get_total(const ht_adwin_t *adwin);

/**
 * @brief Returns the total number of concept drift events detected.
 *
 * @param adwin Pointer to ADWIN detector instance.
 * @return Number of detections.
 */
uint32_t ht_adwin_get_n_detections(const ht_adwin_t *adwin);

/**
 * @brief Computes the analytical ADWIN cut condition epsilon.
 *
 * Evaluates: \epsilon = \sqrt{2 \cdot m \cdot \sigma^2 \cdot \ln(2\ln(n)/\delta)} + \frac{2}{3} \cdot m \cdot \ln(2\ln(n)/\delta)
 *
 * @param n     Window width.
 * @param n0    Subwindow W0 size.
 * @param n1    Subwindow W1 size.
 * @param var   Window sample variance.
 * @param delta Significance parameter.
 * @param min_w Minimum window length parameter.
 * @return Cut threshold epsilon.
 */
ht_float_t ht_adwin_calculate_cut_epsilon(uint32_t n,
                                          uint32_t n0,
                                          uint32_t n1,
                                          ht_float_t var,
                                          ht_float_t delta,
                                          uint16_t min_w);

/* --- ADWIN Pool Allocator API --- */

/**
 * @brief Initializes a deterministic ADWIN detector pool.
 *
 * @param pool Pointer to pool structure.
 * @return HT_OK on success, or HT_ERR_INVALID_ARG if pool is NULL.
 */
ht_status_t ht_adwin_pool_init(ht_adwin_pool_t *pool);

/**
 * @brief Allocates an ADWIN detector from the pool in O(1).
 *
 * @param pool Pointer to pool structure.
 * @param cfg  Configuration parameters (or NULL for defaults).
 * @return Allocated detector index (0..HT_MAX_ADWIN_DETECTORS-1), or HT_INVALID_IDX if full.
 */
uint16_t ht_adwin_pool_alloc(ht_adwin_pool_t *pool, const ht_adwin_config_t *cfg);

/**
 * @brief Frees an ADWIN detector back to the pool in O(1).
 *
 * @param pool Pointer to pool structure.
 * @param idx  Index of detector to release.
 * @return HT_OK on success, or error code on invalid index / double free.
 */
ht_status_t ht_adwin_pool_free(ht_adwin_pool_t *pool, uint16_t idx);

/**
 * @brief Gets a pointer to an allocated ADWIN detector from the pool.
 *
 * @param pool Pointer to pool structure.
 * @param idx  Detector index.
 * @return Pointer to detector, or NULL if index is invalid.
 */
ht_adwin_t *ht_adwin_pool_get(ht_adwin_pool_t *pool, uint16_t idx);

/**
 * @brief Resets all detector allocations in the pool.
 *
 * @param pool Pointer to pool structure.
 */
void ht_adwin_pool_reset(ht_adwin_pool_t *pool);

/**
 * @brief Returns the total RAM bytes utilized by the pool.
 *
 * @param pool Pointer to pool structure.
 * @return Total bytes.
 */
size_t ht_adwin_pool_bytes_used(const ht_adwin_pool_t *pool);

#ifdef __cplusplus
}
#endif

#endif /* HT_DRIFT_H */
