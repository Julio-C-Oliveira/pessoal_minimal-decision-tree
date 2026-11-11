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

#ifndef HT_MAX_DRIFT_NODES
#define HT_MAX_DRIFT_NODES HT_MAX_ADWIN_DETECTORS
#endif

/**
 * @brief Node concept drift monitoring and alternate subtree metadata for HAT.
 */
typedef struct {
    ht_float_t mean_error;   /**< Online error rate / absolute error mean tracked at this node */
    ht_float_t error_m2;     /**< Centered sum of squared errors M2 for bilateral Z-test */
    uint32_t   error_count;  /**< Number of error updates observed */
    uint16_t   adwin_idx;    /**< Index in ht_adwin_pool_t, or HT_INVALID_IDX */
    uint16_t   alt_tree_idx; /**< Root of alternate subtree in node_pool, or HT_INVALID_IDX */
    uint16_t   node_idx;     /**< Owning node index in node_pool */
    uint16_t   next_free;    /**< Intrusive link for free-list when inactive */
} ht_node_drift_t;

/**
 * @brief Dedicated static pool allocator for node drift monitoring structures.
 *
 * Sized at HT_MAX_DRIFT_NODES (128 entries = 2.5 KB in float32 / 4 KB in double64).
 * Replaces the monolithic node_drift[HT_MAX_NODES] array (65.5 KB / 98.3 KB)
 * saving ~63 KB of SRAM in HAT mode and full 65.5 KB in pure VFDT mode.
 */
typedef struct {
    ht_node_drift_t pool[HT_MAX_DRIFT_NODES];
    uint16_t        free_head;
    uint16_t        allocated_count;
} ht_drift_pool_t;

/**
 * @brief Deterministic pool allocator for multiple ADWIN detector instances and drift nodes.
 */
typedef struct {
    ht_adwin_t      pool[HT_MAX_ADWIN_DETECTORS];
    ht_drift_pool_t drift_pool;
    uint16_t        free_indices[HT_MAX_ADWIN_DETECTORS];
    uint32_t        in_use_mask[(HT_MAX_ADWIN_DETECTORS + 31U) / 32U];
    uint16_t        free_top;
    uint16_t        allocated_count;
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

/* --- HAT (Hoeffding Adaptive Tree) Concept Drift Helpers --- */

/**
 * @brief Actions resulting from evaluating an alternate subtree in HAT.
 */
typedef enum {
    HT_HAT_ACTION_NONE   = 0, /**< Keep learning in both trees */
    HT_HAT_ACTION_SWITCH = 1, /**< Alternate tree has significantly lower error; replace main branch */
    HT_HAT_ACTION_PRUNE  = 2  /**< Alternate tree has significantly higher error; prune alternate tree */
} ht_hat_action_t;

/**
 * @brief Computes the statistical bound for alternate tree replacement (Switch/Prune).
 *
 * Evaluates: bound = sqrt(2 * e_old * (1 - e_old) * ln(2 / alpha) * (1/n_alt + 1/n_old))
 *
 * @param e_old Mean error rate of main branch (0.0 .. 1.0).
 * @param n_old Number of observations accumulated in main branch.
 * @param n_alt Number of observations accumulated in alternate tree.
 * @param alpha Significance level (e.g. 0.05).
 * @return Statistical bound (or HT_INFINITY on invalid arguments).
 */
ht_float_t ht_hat_calculate_switch_bound(ht_float_t e_old,
                                         uint32_t n_old,
                                         uint32_t n_alt,
                                         ht_float_t alpha);

/**
 * @brief Evaluates whether an alternate tree should replace the main branch (Switch),
 *        be pruned (Prune), or keep learning (None).
 *
 * @param e_old    Mean error of main branch.
 * @param n_old    Observations in main branch.
 * @param e_alt    Mean error of alternate subtree.
 * @param n_alt    Observations in alternate subtree.
 * @param min_obs  Minimum sample window threshold before decisions (e.g. 300).
 * @param alpha    Significance level (e.g. 0.05).
 * @return HT_HAT_ACTION_SWITCH, HT_HAT_ACTION_PRUNE, or HT_HAT_ACTION_NONE.
 */
ht_hat_action_t ht_hat_evaluate_alternate(ht_float_t e_old,
                                          uint32_t n_old,
                                          ht_float_t e_alt,
                                          uint32_t n_alt,
                                          uint32_t min_obs,
                                          ht_float_t alpha);

/**
 * @brief Unilateral error drift detector for HAT.
 *
 * Updates ADWIN and online error mean with 0-1 loss (detec_in = 0 for correct, 1 for error).
 * If ADWIN detects drift:
 * - If new error is lower than old error: drift represents improving accuracy,
 *   so the signal is filtered out (false alarm) and the error estimator is reset.
 * - If new error is higher or equal: genuine drift detected, returns true.
 *
 * @param adwin Pointer to ADWIN detector.
 * @param mean_error In-out pointer to current mean error.
 * @param error_count In-out pointer to update count.
 * @param detec_in 0 if prediction was correct, 1 if error.
 * @return true if error significantly increased (drift requiring adaptation), false otherwise.
 */
bool ht_hat_check_error_drift(ht_adwin_t *adwin,
                              ht_float_t *mean_error,
                              uint32_t *error_count,
                              uint8_t detec_in);

/**
 * @brief Combines predictions from multiple leaves reached across main and alternate trees,
 *        weighting each leaf's distribution by the inverse squared error (1 / e_l^2).
 *
 * @param leaf_probas Flat array of leaf class distributions of size (n_leaves * HT_MAX_CLASSES).
 * @param mean_errors Array of mean errors for each reached leaf (size n_leaves).
 * @param n_leaves Number of leaves reached.
 * @param n_classes Number of active classes.
 * @param out_proba Output normalized probability array of size HT_MAX_CLASSES.
 */
void ht_hat_combine_leaf_predictions(const ht_float_t *leaf_probas,
                                     const ht_float_t *mean_errors,
                                     uint16_t n_leaves,
                                     uint8_t n_classes,
                                     ht_float_t out_proba[HT_MAX_CLASSES]);

/**
 * @brief Unilateral absolute error drift detector for HAT regression (HAT-R).
 *
 * Updates ADWIN and online error statistics (Welford mean and M2) with absolute prediction error.
 * If ADWIN detects drift:
 * - If current mean error is lower than old mean error: improvement, filtered out (false alarm),
 *   resets error estimator and returns false.
 * - If current mean error is higher or equal: genuine drift detected, resets error estimator
 *   and returns true.
 *
 * @param adwin Pointer to ADWIN detector.
 * @param mean_error In-out pointer to current mean absolute error.
 * @param error_m2 In-out pointer to sum of squared differences from mean error (M2).
 * @param error_count In-out pointer to number of error updates observed.
 * @param abs_error Absolute error observed: |y - y_hat|.
 * @return true if error significantly increased (drift requiring alternate tree), false otherwise.
 */
bool ht_hat_check_reg_error_drift(ht_adwin_t *adwin,
                                  ht_float_t *mean_error,
                                  ht_float_t *error_m2,
                                  uint32_t *error_count,
                                  ht_float_t abs_error);

/**
 * @brief Evaluates an alternate subtree against the main branch in HAT-R using a two-tailed Z-test.
 *
 * Implements Welch-Z test matching River's hatr_nodes.py:
 * z = (e_alt - e_old) / sqrt(var_alt / n_alt + var_old / n_old)
 * p_value = erfc(|z| / sqrt(2))
 *
 * If p_value <= alpha:
 * - If e_alt < e_old: HT_HAT_ACTION_SWITCH
 * - Else (e_alt >= e_old): HT_HAT_ACTION_PRUNE
 * Else: HT_HAT_ACTION_NONE
 *
 * @param e_old Mean absolute error of main branch.
 * @param var_old Error variance (sample s^2) of main branch.
 * @param n_old Number of samples observed in main branch.
 * @param e_alt Mean absolute error of alternate subtree.
 * @param var_alt Error variance (sample s^2) of alternate subtree.
 * @param n_alt Number of samples observed in alternate subtree.
 * @param min_obs Minimum sample window threshold before decisions (e.g. 300).
 * @param alpha Significance level (e.g. 0.05).
 * @return HT_HAT_ACTION_SWITCH, HT_HAT_ACTION_PRUNE, or HT_HAT_ACTION_NONE.
 */
ht_hat_action_t ht_hat_evaluate_reg_alternate(ht_float_t e_old,
                                              ht_float_t var_old,
                                              uint32_t n_old,
                                              ht_float_t e_alt,
                                              ht_float_t var_alt,
                                              uint32_t n_alt,
                                              uint32_t min_obs,
                                              ht_float_t alpha);

/* --- Public Drift Pool API --- */

/**
 * @brief Initializes a dedicated static drift pool with intrusive free-list.
 */
ht_status_t ht_drift_pool_init(ht_drift_pool_t *pool);

/**
 * @brief Allocates an entry from the drift pool in O(1) time.
 *
 * @return Allocated entry index, or HT_INVALID_IDX if pool is exhausted.
 */
uint16_t ht_drift_pool_alloc(ht_drift_pool_t *pool);

/**
 * @brief Releases a drift pool entry in O(1) time.
 */
ht_status_t ht_drift_pool_free(ht_drift_pool_t *pool, uint16_t idx);

/**
 * @brief Retrieves a mutable pointer to a drift pool entry.
 */
ht_node_drift_t *ht_drift_pool_get(ht_drift_pool_t *pool, uint16_t idx);

/**
 * @brief Retrieves a const pointer to a drift pool entry.
 */
const ht_node_drift_t *ht_drift_pool_get_const(const ht_drift_pool_t *pool, uint16_t idx);

/**
 * @brief Resets all entries in the drift pool back to their free state.
 */
void ht_drift_pool_reset(ht_drift_pool_t *pool);

/**
 * @brief Returns total bytes occupied by the drift pool.
 */
size_t ht_drift_pool_bytes_used(const ht_drift_pool_t *pool);

#ifdef __cplusplus
}
#endif

#endif /* HT_DRIFT_H */
