/**
 * @file ht_observer.h
 * @brief Attribute observers (Gaussian Welford and Nominal Categorical) for embedded VFDT / HAT.
 *
 * Implements incremental feature statistics tracking with O(1) time and strict O(C) memory:
 * - GaussianSplitter: Welford's algorithm tracking (count, mean, M2, min, max) per class.
 *   Provides analytic normal PDF, log-PDF, CDF, candidate split suggestions, and
 *   post-split class distribution estimation.
 * - NominalSplitterClassif: 2D frequency matrix (classes x categories) supporting
 *   one-vs-rest binary split distribution estimation and conditional probabilities.
 * - Stack-allocated ht_split_candidate_t replacing River's heap-allocated BranchFactory.
 *
 * Strictly adheres to the zero-malloc rule and <= 1 MB RAM budget.
 */

#ifndef HT_OBSERVER_H
#define HT_OBSERVER_H

#include "ht_config.h"
#include "ht_memory.h"

/**
 * @brief Stack-allocated candidate split descriptor (replaces River's heap-allocated BranchFactory).
 *
 * Stores evaluated impurity merit and estimated post-split child class distributions.
 * Layout ordered by decreasing field alignment to eliminate wasteful internal padding:
 * - 4 or 8 bytes: merit, lhs_dist, rhs_dist, split_val
 * - 1 byte: feature_idx, is_categorical
 *
 * Exactly 44 bytes in float32 and 88 bytes in float64.
 */
typedef struct {
    ht_float_t merit;                            /* Impurity merit (-HT_INFINITY if invalid) */
    ht_float_t lhs_dist[HT_MAX_CLASSES];         /* Estimated distribution for left branch */
    ht_float_t rhs_dist[HT_MAX_CLASSES];         /* Estimated distribution for right branch */
    union {
        ht_float_t threshold;                    /* Numeric split threshold */
        int32_t    cat_val;                      /* Categorical equality value */
    } split_val;
    uint8_t    feature_idx;                      /* Feature index (HT_NO_FEATURE for null split) */
    uint8_t    is_categorical;                   /* 0 = numeric, 1 = categorical */
} ht_split_candidate_t;

/* Compile-time static assertions for exact memory layout */
#ifndef HT_USE_DOUBLE
_Static_assert(sizeof(ht_split_candidate_t) == 44, "ht_split_candidate_t must be 44 bytes in float32 mode");
#else
_Static_assert(sizeof(ht_split_candidate_t) == 88, "ht_split_candidate_t must be 88 bytes in float64 mode");
#endif

/* ========================================================================= */
/* --- Gaussian Numeric Observer API (Welford's Algorithm) ----------------- */
/* ========================================================================= */

/**
 * @brief Initializes a single Gaussian observer to zero state.
 *
 * @param obs Pointer to the Gaussian observer.
 */
void ht_gaussian_init(ht_gaussian_obs_t *obs);

/**
 * @brief Incremental update using Welford's algorithm for 1 (feature, class) pair.
 *
 * Updates sample weight count (n_c), mean (mu_c), centered sum of squares (M2_c),
 * and observed range [min_val, max_val]. Safely ignores NaN values, weights <= 0,
 * and class indices >= HT_MAX_CLASSES.
 *
 * @param obs Array of Gaussian observers for all classes of a single feature.
 * @param class_idx Target class index (0 .. HT_MAX_CLASSES - 1).
 * @param val Feature value (ignored if NaN).
 * @param weight Sample weight (ignored if <= 0).
 */
void ht_gaussian_update(ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                        uint8_t class_idx,
                        ht_float_t val,
                        ht_float_t weight);

/**
 * @brief Computes unbiased sample variance s^2 = M2 / (n - 1) for n > 1.
 *
 * Returns 0.0 if n <= 1 or M2 <= 0.
 *
 * @param obs Pointer to the Gaussian observer.
 * @return Sample variance s^2.
 */
ht_float_t ht_gaussian_variance(const ht_gaussian_obs_t *obs);

/**
 * @brief Computes sample standard deviation sigma = sqrt(s^2).
 *
 * Returns 0.0 if n <= 1 or s^2 <= 0.
 *
 * @param obs Pointer to the Gaussian observer.
 * @return Standard deviation sigma.
 */
ht_float_t ht_gaussian_stddev(const ht_gaussian_obs_t *obs);

/**
 * @brief Evaluates the standard normal cumulative distribution function (CDF).
 *
 * Computes Phi((x - mu) / sigma) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2)))).
 * Degenerate cases: if sigma <= 0, returns 0.0 (x < mu), 1.0 (x > mu), or 0.5 (x == mu).
 *
 * @param x Evaluation point.
 * @param mean Distribution mean mu.
 * @param stddev Distribution standard deviation sigma.
 * @return Probability Phi((x - mu) / sigma) in [0.0, 1.0].
 */
ht_float_t ht_normal_cdf(ht_float_t x, ht_float_t mean, ht_float_t stddev);

/**
 * @brief Evaluates Gaussian probability density function (PDF).
 *
 * Computes p(x | c) = 1 / sqrt(2 * pi * s^2) * exp(-(x - mu)^2 / (2 * s^2)).
 * Returns 0.0 if count < 2 or variance <= 0.
 *
 * @param obs Pointer to the Gaussian observer.
 * @param val Feature value.
 * @return Density p(x | c).
 */
ht_float_t ht_gaussian_cond_proba(const ht_gaussian_obs_t *obs, ht_float_t val);

/**
 * @brief Evaluates Gaussian log probability density function (log-PDF).
 *
 * Computes ln(p(x | c)) = -0.5 * (ln(2 * pi) + ln(s^2)) - (x - mu)^2 / (2 * s^2).
 * Returns -HT_INFINITY if count < 2 or variance <= 0 or val is NaN.
 *
 * @param obs Pointer to the Gaussian observer.
 * @param val Feature value.
 * @return Log density ln(p(x | c)).
 */
ht_float_t ht_gaussian_cond_log_proba(const ht_gaussian_obs_t *obs, ht_float_t val);

/**
 * @brief Generates up to HT_GAUSSIAN_SPLITS evenly spaced candidate split points.
 *
 * Matches River's GaussianSplitter._split_point_suggestions:
 * Finds [min_val, max_val] across all observed classes with count > 0,
 * divides into HT_GAUSSIAN_SPLITS + 1 bins, and outputs interior candidate points.
 * Returns 0 if feature is constant (min_val >= max_val) or no samples observed.
 *
 * @param obs Array of Gaussian observers for all classes of a feature.
 * @param n_classes Number of classes in the tree schema.
 * @param splits_out Buffer of at least HT_GAUSSIAN_SPLITS elements to receive suggested points.
 * @return Number of candidate split points written to splits_out.
 */
uint8_t ht_gaussian_split_suggestions(const ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                                      uint8_t n_classes,
                                      ht_float_t splits_out[HT_GAUSSIAN_SPLITS]);

/**
 * @brief Estimates post-split class distributions (lhs_dist, rhs_dist) for a numeric split.
 *
 * Matches River's GaussianSplitter._class_dists_from_binary_split:
 * For each class c:
 * - If split_val < min_val: all weight goes to right branch.
 * - If split_val >= max_val: all weight goes to left branch.
 * - Otherwise: left_weight = cdf(split_val) * count, right_weight = count - left_weight.
 *
 * @param obs Array of Gaussian observers for all classes.
 * @param n_classes Number of active classes.
 * @param split_val Candidate threshold.
 * @param lhs_dist Output array of size HT_MAX_CLASSES for left branch (x <= split_val).
 * @param rhs_dist Output array of size HT_MAX_CLASSES for right branch (x > split_val).
 */
void ht_gaussian_binary_split_dists(const ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                                    uint8_t n_classes,
                                    ht_float_t split_val,
                                    ht_float_t lhs_dist[HT_MAX_CLASSES],
                                    ht_float_t rhs_dist[HT_MAX_CLASSES]);

/* ========================================================================= */
/* --- Categorical Nominal Observer API (One-vs-Rest) ---------------------- */
/* ========================================================================= */

/**
 * @brief Initializes a categorical nominal observer to zero state.
 *
 * @param obs Pointer to categorical observer.
 */
void ht_categorical_init(ht_categorical_obs_t *obs);

/**
 * @brief Increments frequency count for a given class and nominal category.
 *
 * Safely ignores invalid category (< 0 or >= HT_MAX_CATEGORIES), invalid class,
 * or non-positive weight.
 *
 * @param obs Pointer to categorical observer.
 * @param class_idx Target class index.
 * @param cat_val Discrete category value (0 .. HT_MAX_CATEGORIES - 1).
 * @param weight Sample weight.
 */
void ht_categorical_update(ht_categorical_obs_t *obs,
                           uint8_t class_idx,
                           int32_t cat_val,
                           ht_float_t weight);

/**
 * @brief Computes conditional probability P(x == cat_val | class_idx).
 *
 * Returns N_{v, c} / sum_u N_{u, c}. Returns 0.0 if total class weight is 0
 * or category is invalid.
 *
 * @param obs Pointer to categorical observer.
 * @param class_idx Target class index.
 * @param cat_val Discrete category value.
 * @return Conditional probability in [0.0, 1.0].
 */
ht_float_t ht_categorical_cond_proba(const ht_categorical_obs_t *obs,
                                     uint8_t class_idx,
                                     int32_t cat_val);

/**
 * @brief Checks if a category value was observed at least once across any class.
 *
 * @param obs Pointer to categorical observer.
 * @param cat_val Discrete category value.
 * @return true if observed with weight > 0, false otherwise.
 */
bool ht_categorical_is_observed(const ht_categorical_obs_t *obs, int32_t cat_val);

/**
 * @brief Computes binary post-split class distributions (one-vs-rest: x == cat_val vs x != cat_val).
 *
 * Matches River's NominalSplitterClassif._class_dist_from_binary_split:
 * - Left branch (x == cat_val): lhs_dist[c] = counts[c][cat_val]
 * - Right branch (x != cat_val): rhs_dist[c] = sum_{u != cat_val} counts[c][u]
 *
 * @param obs Pointer to categorical observer.
 * @param n_classes Number of active classes.
 * @param cat_val Discrete category value.
 * @param lhs_dist Output array of size HT_MAX_CLASSES for equality branch (x == cat_val).
 * @param rhs_dist Output array of size HT_MAX_CLASSES for rest branch (x != cat_val).
 */
void ht_categorical_binary_split_dists(const ht_categorical_obs_t *obs,
                                       uint8_t n_classes,
                                       int32_t cat_val,
                                       ht_float_t lhs_dist[HT_MAX_CLASSES],
                                       ht_float_t rhs_dist[HT_MAX_CLASSES]);

/* ========================================================================= */
/* --- Candidate Helper ---------------------------------------------------- */
/* ========================================================================= */

/**
 * @brief Initializes a split candidate descriptor to empty / invalid state.
 *
 * Sets merit = -HT_INFINITY, feature_idx = HT_NO_FEATURE, and clears distributions.
 *
 * @param cand Pointer to candidate descriptor.
 */
void ht_split_candidate_init(ht_split_candidate_t *cand);

#endif /* HT_OBSERVER_H */
