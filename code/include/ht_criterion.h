/**
 * @file ht_criterion.h
 * @brief Split criteria (Gini Impurity & Information Gain) and Hoeffding Bound evaluation.
 *
 * Implements tree-level polymorphism via ht_criterion_vtable_t without dynamic allocation:
 * - Gini Impurity: 1.0 - sum_b (W_b / W_post) * Gini(D_b), with range_of_merit = 1.0.
 * - Information Gain: H(D_pre) - sum_b (W_b / W_post) * H(D_b), with single-division
 *   Shannon entropy in base 2, range_of_merit = log2(max(C_obs, 2)).
 * - Hoeffding Bound: epsilon = sqrt((R^2 * ln(1 / delta)) / (2 * n)).
 * - Evaluation of numeric (Gaussian) and categorical (nominal) observers.
 * - Selection of best and second-best candidate splits with stable River-matching tie-breaking.
 * - Hoeffding tree split decision logic with tie-breaking parameter tau.
 *
 * Strictly adheres to C11 (-std=c11 -Wall -Wextra -Werror -pedantic) and <= 1 MB RAM budget.
 */

#ifndef HT_CRITERION_H
#define HT_CRITERION_H

#include "ht_config.h"
#include "ht_memory.h"
#include "ht_observer.h"

/**
 * @brief Supported split criteria identifiers.
 */
typedef enum {
    HT_CRIT_INFO_GAIN = 0,
    HT_CRIT_GINI      = 1
} ht_criterion_type_t;

/**
 * @brief Virtual table defining split criterion operations.
 */
typedef struct {
    /**
     * @brief Computes range of merit R for the Hoeffding bound.
     *
     * For Gini: R = 1.0.
     * For Information Gain: R = log2(max(C_obs, 2)), where C_obs is the number of
     * classes observed in pre_dist with weight > 0.
     *
     * @param pre_dist Class distribution prior to split.
     * @param n_classes Number of classes in schema.
     * @return Range of merit R.
     */
    ht_float_t (*range_of_merit)(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                 uint8_t n_classes);

    /**
     * @brief Computes impurity merit of a proposed binary split.
     *
     * Enforces min_branch_fraction: returns -HT_INFINITY if either branch fraction
     * (W_b / W_post) is <= min_branch_fraction or if W_post <= 0.
     *
     * @param pre_dist Class distribution prior to split.
     * @param lhs_dist Class distribution for left branch (x <= val or x == cat).
     * @param rhs_dist Class distribution for right branch (x > val or x != cat).
     * @param n_classes Number of classes in schema.
     * @param min_branch_fraction Minimum allowed weight fraction per branch.
     * @return Impurity merit (higher is better), or -HT_INFINITY if invalid.
     */
    ht_float_t (*merit_of_split)(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                 const ht_float_t lhs_dist[HT_MAX_CLASSES],
                                 const ht_float_t rhs_dist[HT_MAX_CLASSES],
                                 uint8_t n_classes,
                                 ht_float_t min_branch_fraction);
} ht_criterion_vtable_t;

/* ========================================================================= */
/* --- Criterion Impurity and Entropy Functions ---------------------------- */
/* ========================================================================= */

/**
 * @brief Computes Gini impurity of a class distribution.
 *
 * Gini(D) = 1.0 - sum_{c: w_c > 0} (w_c / W)^2.
 * Returns 1.0 if dist_sum <= 0.
 *
 * @param dist Class distribution array of size HT_MAX_CLASSES.
 * @param n_classes Number of classes in schema.
 * @param dist_sum Total sum of weights in dist (passed to avoid recomputing).
 * @return Gini impurity in [0.0, 1.0].
 */
ht_float_t ht_compute_gini(const ht_float_t dist[HT_MAX_CLASSES],
                           uint8_t n_classes,
                           ht_float_t dist_sum);

/**
 * @brief Computes Shannon entropy in base 2 of a class distribution using single division.
 *
 * H(D) = (dis_sums * log2(dis_sums) - sum_{c: w_c > 0} w_c * log2(w_c)) / dis_sums.
 * Returns 0.0 if dis_sums <= 0 or distribution is pure.
 *
 * @param dist Class distribution array of size HT_MAX_CLASSES.
 * @param n_classes Number of classes in schema.
 * @return Shannon entropy in base 2 (>= 0.0).
 */
ht_float_t ht_compute_entropy(const ht_float_t dist[HT_MAX_CLASSES],
                              uint8_t n_classes);

/**
 * @brief Retrieves criterion vtable pointer for the given criterion type.
 *
 * @param type HT_CRIT_INFO_GAIN or HT_CRIT_GINI.
 * @return Pointer to static const vtable, or NULL if invalid type.
 */
const ht_criterion_vtable_t *ht_criterion_get_vtable(ht_criterion_type_t type);

/* ========================================================================= */
/* --- Hoeffding Bound & Split Evaluation ---------------------------------- */
/* ========================================================================= */

/**
 * @brief Computes Hoeffding bound epsilon.
 *
 * epsilon = sqrt((range_val^2 * ln(1 / delta)) / (2 * total_weight)).
 * Returns HT_INFINITY if total_weight <= 0 or delta <= 0.
 * Returns 0.0 if delta >= 1.0.
 *
 * @param range_val Range of merit R.
 * @param delta Desired split confidence error bound (e.g. 1e-7).
 * @param total_weight Accumulated weight seen at the node.
 * @return Hoeffding bound epsilon.
 */
ht_float_t ht_hoeffding_bound(ht_float_t range_val,
                              ht_float_t delta,
                              ht_float_t total_weight);

/**
 * @brief Evaluates numeric Gaussian observer to find best split threshold.
 *
 * Generates suggested split points from observer, computes post-split distributions,
 * evaluates merit with criterion, and writes best candidate.
 * If all splits violate min_branch_fraction or feature is constant, candidate
 * will have merit = -HT_INFINITY and feature_idx = HT_NO_FEATURE.
 * Ties within the feature are broken by preferring the earlier (lower) threshold.
 *
 * @param obs Gaussian observers for all classes of a numeric feature.
 * @param pre_dist Class distribution prior to split.
 * @param n_classes Number of classes in schema.
 * @param feature_idx Feature index in tree schema.
 * @param crit Criterion vtable.
 * @param min_branch_frac Minimum fraction of total weight required in each child.
 * @param out_best Output candidate descriptor.
 */
void ht_evaluate_numeric_observer(const ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                                  const ht_float_t pre_dist[HT_MAX_CLASSES],
                                  uint8_t n_classes,
                                  uint8_t feature_idx,
                                  const ht_criterion_vtable_t *crit,
                                  ht_float_t min_branch_frac,
                                  ht_split_candidate_t *out_best);

/**
 * @brief Evaluates categorical nominal observer to find best one-vs-rest split.
 *
 * Evaluates each observed category value as a binary split candidate (x == cat vs x != cat).
 * If all splits violate min_branch_fraction or no categories observed, candidate
 * will have merit = -HT_INFINITY and feature_idx = HT_NO_FEATURE.
 * Ties within the feature are broken by preferring the earlier category.
 *
 * @param obs Categorical observer for nominal feature.
 * @param pre_dist Class distribution prior to split.
 * @param n_classes Number of classes in schema.
 * @param feature_idx Feature index in tree schema.
 * @param crit Criterion vtable.
 * @param min_branch_frac Minimum fraction of total weight required in each child.
 * @param out_best Output candidate descriptor.
 */
void ht_evaluate_categorical_observer(const ht_categorical_obs_t *obs,
                                      const ht_float_t pre_dist[HT_MAX_CLASSES],
                                      uint8_t n_classes,
                                      uint8_t feature_idx,
                                      const ht_criterion_vtable_t *crit,
                                      ht_float_t min_branch_frac,
                                      ht_split_candidate_t *out_best);

/**
 * @brief Evaluates all feature observers in an active leaf and returns top 2 split candidates.
 *
 * Matches River's leaf.best_split_suggestions():
 * - If merit_preprune is true, null split candidate is added first.
 * - Each feature observer is evaluated.
 * - Candidates are stably sorted by merit (River Timsort parity: on merit tie, higher feature index wins).
 * - Writes best candidate to out_best and second best to out_second_best.
 *
 * @param stats Pointer to leaf statistics block.
 * @param pre_dist Class distribution prior to split (node->class_counts).
 * @param feature_types Array of feature types for the tree schema.
 * @param n_features Number of features in schema.
 * @param n_classes Number of classes in schema.
 * @param crit Criterion vtable.
 * @param min_branch_frac Minimum fraction of total weight required in each child.
 * @param merit_preprune Whether to include null split as candidate.
 * @param out_best Output for top split candidate.
 * @param out_second_best Output for second best candidate (or invalid if < 2 candidates).
 */
void ht_find_best_split_candidates(const ht_leaf_stats_t *stats,
                                   const ht_float_t pre_dist[HT_MAX_CLASSES],
                                   const ht_feature_type_t feature_types[HT_MAX_FEATURES],
                                   uint8_t n_features,
                                   uint8_t n_classes,
                                   const ht_criterion_vtable_t *crit,
                                   ht_float_t min_branch_frac,
                                   bool merit_preprune,
                                   ht_split_candidate_t *out_best,
                                   ht_split_candidate_t *out_second_best);

/**
 * @brief Decides whether to perform a split based on Hoeffding bound and tie threshold.
 *
 * A split is triggered if:
 * 1. best candidate has valid merit (> -HT_INFINITY) and valid feature_idx != HT_NO_FEATURE.
 * 2. And either:
 *    - second_best is invalid / -HT_INFINITY (only 1 candidate).
 *    - OR (best->merit - second_best->merit) > hoeffding_bound.
 *    - OR hoeffding_bound < tau (tie-breaker for very similar attributes).
 *
 * @param best Top split candidate.
 * @param second_best Second best candidate.
 * @param hoeffding_bound Computed Hoeffding bound epsilon.
 * @param tau Tie-breaker threshold (e.g. 0.05).
 * @return true if split condition is met, false otherwise.
 */
bool ht_should_split(const ht_split_candidate_t *best,
                     const ht_split_candidate_t *second_best,
                     ht_float_t hoeffding_bound,
                     ht_float_t tau);

#endif /* HT_CRITERION_H */
