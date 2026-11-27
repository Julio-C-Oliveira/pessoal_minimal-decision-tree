/**
 * @file ht_criterion.c
 * @brief Implementation of split criteria (Gini, InfoGain, Variance Reduction)
 *        and Hoeffding bound evaluation for float32, double64, and Q16.16 fixed-point.
 */

#include "ht_criterion.h"

/* ========================================================================= */
/* --- Gini Impurity Criterion --------------------------------------------- */
/* ========================================================================= */

ht_float_t ht_compute_gini(const ht_float_t dist[HT_MAX_CLASSES],
                           uint8_t n_classes,
                           ht_float_t dist_sum)
{
    ht_float_t gini = HT_CONST(1.0);
    if (dist_sum > HT_CONST(0.0)) {
        for (uint8_t c = 0; c < n_classes; c++) {
            if (dist[c] > HT_CONST(0.0)) {
                ht_float_t rel_freq = HT_DIV(dist[c], dist_sum);
                gini = HT_SUB(gini, HT_MUL(rel_freq, rel_freq));
            }
        }
    }
    return gini;
}

static ht_float_t ht_gini_range_of_merit(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                         uint8_t n_classes)
{
    (void)pre_dist;
    (void)n_classes;
    return HT_CONST(1.0);
}

static ht_float_t ht_gini_merit_of_split(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                         const ht_float_t lhs_dist[HT_MAX_CLASSES],
                                         const ht_float_t rhs_dist[HT_MAX_CLASSES],
                                         uint8_t n_classes,
                                         ht_float_t min_branch_fraction)
{
    (void)pre_dist;
#ifdef HT_USE_FIXED_POINT
    int64_t w_l64 = 0;
    int64_t w_r64 = 0;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (lhs_dist[c] > 0) w_l64 += (int64_t)lhs_dist[c];
        if (rhs_dist[c] > 0) w_r64 += (int64_t)rhs_dist[c];
    }
    int64_t total64 = w_l64 + w_r64;
    if (total64 <= 0) {
        return -HT_INFINITY;
    }
    ht_float_t frac_l = (ht_float_t)((((uint64_t)w_l64 << 16) + ((uint64_t)total64 >> 1)) / (uint64_t)total64);
    ht_float_t frac_r = (ht_float_t)((((uint64_t)w_r64 << 16) + ((uint64_t)total64 >> 1)) / (uint64_t)total64);
    if (frac_l <= min_branch_fraction || frac_r <= min_branch_fraction) {
        return -HT_INFINITY;
    }
    /* Compute branch Ginis in 64-bit denominator to avoid saturation on large sums */
    ht_float_t gini_l = HT_FIXED_ONE;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (lhs_dist[c] > 0) {
            ht_float_t p = (ht_float_t)((((uint64_t)(uint32_t)lhs_dist[c] << 16) + ((uint64_t)w_l64 >> 1)) / (uint64_t)w_l64);
            gini_l = HT_SUB(gini_l, HT_MUL(p, p));
        }
    }
    ht_float_t gini_r = HT_FIXED_ONE;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (rhs_dist[c] > 0) {
            ht_float_t p = (ht_float_t)((((uint64_t)(uint32_t)rhs_dist[c] << 16) + ((uint64_t)w_r64 >> 1)) / (uint64_t)w_r64);
            gini_r = HT_SUB(gini_r, HT_MUL(p, p));
        }
    }
    ht_float_t gini = HT_ADD(HT_MUL(frac_l, gini_l), HT_MUL(frac_r, gini_r));
    return HT_SUB(HT_FIXED_ONE, gini);
#else
    ht_float_t w_l = HT_CONST(0.0);
    ht_float_t w_r = HT_CONST(0.0);

    for (uint8_t c = 0; c < n_classes; c++) {
        if (lhs_dist[c] > HT_CONST(0.0)) {
            w_l = HT_ADD(w_l, lhs_dist[c]);
        }
        if (rhs_dist[c] > HT_CONST(0.0)) {
            w_r = HT_ADD(w_r, rhs_dist[c]);
        }
    }

    ht_float_t total_weight = HT_ADD(w_l, w_r);
    if (total_weight <= HT_CONST(0.0)) {
        return -HT_INFINITY;
    }

    if (HT_DIV(w_l, total_weight) <= min_branch_fraction ||
        HT_DIV(w_r, total_weight) <= min_branch_fraction) {
        return -HT_INFINITY;
    }

    ht_float_t gini = HT_ADD(HT_MUL(HT_DIV(w_l, total_weight), ht_compute_gini(lhs_dist, n_classes, w_l)),
                             HT_MUL(HT_DIV(w_r, total_weight), ht_compute_gini(rhs_dist, n_classes, w_r)));

    return HT_SUB(HT_CONST(1.0), gini);
#endif
}

static const ht_criterion_vtable_t s_gini_vtable = {
    .range_of_merit = ht_gini_range_of_merit,
    .merit_of_split = ht_gini_merit_of_split
};

/* ========================================================================= */
/* --- Information Gain Criterion ------------------------------------------ */
/* ========================================================================= */

ht_float_t ht_compute_entropy(const ht_float_t dist[HT_MAX_CLASSES],
                              uint8_t n_classes)
{
#ifdef HT_USE_FIXED_POINT
    int64_t sum64 = 0;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (dist[c] > 0) {
            sum64 += (int64_t)dist[c];
        }
    }
    if (sum64 <= 0) {
        return 0;
    }
    /* Normalize p_c = w_c / W in [0, 1] so p_c * log2(p_c) never overflows Q16.16 */
    ht_float_t entropy = 0;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (dist[c] > 0) {
            ht_float_t p = (ht_float_t)((((uint64_t)(uint32_t)dist[c] << 16) + ((uint64_t)sum64 >> 1)) / (uint64_t)sum64);
            if (p > 0 && p < HT_FIXED_ONE) {
                entropy = HT_SUB(entropy, HT_MUL(p, HT_LOG2(p)));
            }
        }
    }
    return (entropy > 0) ? entropy : 0;
#else
    ht_float_t entropy = HT_CONST(0.0);
    ht_float_t dis_sums = HT_CONST(0.0);

    for (uint8_t c = 0; c < n_classes; c++) {
        if (dist[c] > HT_CONST(0.0)) {
            entropy = HT_SUB(entropy, HT_MUL(dist[c], HT_LOG2(dist[c])));
            dis_sums = HT_ADD(dis_sums, dist[c]);
        }
    }

    if (dis_sums > HT_CONST(0.0)) {
        return HT_DIV(HT_ADD(entropy, HT_MUL(dis_sums, HT_LOG2(dis_sums))), dis_sums);
    }

    return HT_CONST(0.0);
#endif
}

static ht_float_t ht_info_gain_range_of_merit(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                              uint8_t n_classes)
{
    uint8_t num_observed = 0;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (pre_dist[c] > HT_CONST(0.0)) {
            num_observed++;
        }
    }

    uint8_t k = (num_observed > 2) ? num_observed : 2;
    return HT_LOG2(HT_FROM_UINT(k));
}

static ht_float_t ht_info_gain_merit_of_split(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                              const ht_float_t lhs_dist[HT_MAX_CLASSES],
                                              const ht_float_t rhs_dist[HT_MAX_CLASSES],
                                              uint8_t n_classes,
                                              ht_float_t min_branch_fraction)
{
#ifdef HT_USE_FIXED_POINT
    int64_t w_l64 = 0;
    int64_t w_r64 = 0;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (lhs_dist[c] > 0) w_l64 += (int64_t)lhs_dist[c];
        if (rhs_dist[c] > 0) w_r64 += (int64_t)rhs_dist[c];
    }
    int64_t total64 = w_l64 + w_r64;
    if (total64 <= 0) {
        return -HT_INFINITY;
    }
    ht_float_t frac_l = (ht_float_t)((((uint64_t)w_l64 << 16) + ((uint64_t)total64 >> 1)) / (uint64_t)total64);
    ht_float_t frac_r = (ht_float_t)((((uint64_t)w_r64 << 16) + ((uint64_t)total64 >> 1)) / (uint64_t)total64);
    if (frac_l <= min_branch_fraction || frac_r <= min_branch_fraction) {
        return -HT_INFINITY;
    }
    ht_float_t post_entropy = HT_ADD(HT_MUL(frac_l, ht_compute_entropy(lhs_dist, n_classes)),
                                     HT_MUL(frac_r, ht_compute_entropy(rhs_dist, n_classes)));
    ht_float_t pre_entropy = ht_compute_entropy(pre_dist, n_classes);
    return HT_SUB(pre_entropy, post_entropy);
#else
    ht_float_t w_l = HT_CONST(0.0);
    ht_float_t w_r = HT_CONST(0.0);

    for (uint8_t c = 0; c < n_classes; c++) {
        if (lhs_dist[c] > HT_CONST(0.0)) {
            w_l = HT_ADD(w_l, lhs_dist[c]);
        }
        if (rhs_dist[c] > HT_CONST(0.0)) {
            w_r = HT_ADD(w_r, rhs_dist[c]);
        }
    }

    ht_float_t total_weight = HT_ADD(w_l, w_r);
    if (total_weight <= HT_CONST(0.0)) {
        return -HT_INFINITY;
    }

    if (HT_DIV(w_l, total_weight) <= min_branch_fraction ||
        HT_DIV(w_r, total_weight) <= min_branch_fraction) {
        return -HT_INFINITY;
    }

    ht_float_t post_entropy = HT_DIV(
        HT_ADD(HT_MUL(w_l, ht_compute_entropy(lhs_dist, n_classes)),
               HT_MUL(w_r, ht_compute_entropy(rhs_dist, n_classes))),
        total_weight);
    ht_float_t pre_entropy = ht_compute_entropy(pre_dist, n_classes);

    return HT_SUB(pre_entropy, post_entropy);
#endif
}

static const ht_criterion_vtable_t s_info_gain_vtable = {
    .range_of_merit = ht_info_gain_range_of_merit,
    .merit_of_split = ht_info_gain_merit_of_split
};

/* ========================================================================= */
/* --- Variance Reduction Criterion Vtable --------------------------------- */
/* ========================================================================= */

static ht_float_t ht_vr_range_of_merit(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                       uint8_t n_classes)
{
    (void)pre_dist;
    (void)n_classes;
    return HT_CONST(1.0);
}

static ht_float_t ht_vr_merit_of_split(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                       const ht_float_t lhs_dist[HT_MAX_CLASSES],
                                       const ht_float_t rhs_dist[HT_MAX_CLASSES],
                                       uint8_t n_classes,
                                       ht_float_t min_branch_fraction)
{
    (void)pre_dist;
    (void)lhs_dist;
    (void)rhs_dist;
    (void)n_classes;
    (void)min_branch_fraction;
    return -HT_INFINITY;
}

static const ht_criterion_vtable_t s_variance_reduction_vtable = {
    .range_of_merit = ht_vr_range_of_merit,
    .merit_of_split = ht_vr_merit_of_split
};

/* ========================================================================= */
/* --- Public API ---------------------------------------------------------- */
/* ========================================================================= */

const ht_criterion_vtable_t *ht_criterion_get_vtable(ht_criterion_type_t type)
{
    switch (type) {
        case HT_CRIT_GINI:
            return &s_gini_vtable;
        case HT_CRIT_INFO_GAIN:
            return &s_info_gain_vtable;
        case HT_CRIT_VARIANCE_REDUCTION:
            return &s_variance_reduction_vtable;
        default:
            return NULL;
    }
}

ht_float_t ht_hoeffding_bound_ln(ht_float_t range_val,
                                 ht_float_t neg_ln_delta,
                                 ht_float_t total_weight)
{
    if (total_weight <= HT_CONST(0.0) || neg_ln_delta < HT_CONST(0.0)) {
        return HT_INFINITY;
    }
    if (neg_ln_delta == HT_CONST(0.0)) {
        return HT_CONST(0.0);
    }
#ifdef HT_USE_FIXED_POINT
    if (!HT_ISFINITE(range_val) || !HT_ISFINITE(neg_ln_delta)) {
        return HT_INFINITY;
    }
    uint64_t ar = (range_val < 0) ? (uint64_t)(-(int64_t)range_val) : (uint64_t)range_val;
    uint64_t r2_q16 = (ar * ar + 32768ULL) >> 16;
    uint64_t num_q16 = (r2_q16 * (uint64_t)(uint32_t)neg_ln_delta + 32768ULL) >> 16;
    uint64_t denom = 2ULL * (uint64_t)(uint32_t)total_weight;
    uint64_t radicand_q16;
    if (num_q16 <= (UINT64_MAX >> 16)) {
        radicand_q16 = ((num_q16 << 16) + (denom >> 1)) / denom;
    } else {
        radicand_q16 = (num_q16 / denom) << 16;
    }
    if (radicand_q16 >= (uint64_t)INT32_MAX) {
        return HT_SQRT(INT32_MAX);
    }
    return HT_SQRT((ht_float_t)radicand_q16);
#else
    return HT_SQRT(HT_DIV(HT_MUL(HT_MUL(range_val, range_val), neg_ln_delta),
                          HT_MUL(HT_CONST(2.0), total_weight)));
#endif
}

ht_float_t ht_hoeffding_bound(ht_float_t range_val,
                              ht_float_t delta,
                              ht_float_t total_weight)
{
    if (total_weight <= HT_CONST(0.0) || delta <= HT_CONST(0.0)) {
        return HT_INFINITY;
    }
    if (delta >= HT_CONST(1.0)) {
        return HT_CONST(0.0);
    }
#ifdef HT_USE_FIXED_POINT
    /* Use -ln(delta) directly to avoid 1/delta overflow for small delta */
    ht_float_t neg_ln_delta = -HT_LOG(delta);
    return ht_hoeffding_bound_ln(range_val, neg_ln_delta, total_weight);
#else
    return HT_SQRT(HT_DIV(HT_MUL(HT_MUL(range_val, range_val), HT_LOG(HT_DIV(HT_CONST(1.0), delta))),
                          HT_MUL(HT_CONST(2.0), total_weight)));
#endif
}

void ht_evaluate_numeric_observer(const ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                                  const ht_float_t pre_dist[HT_MAX_CLASSES],
                                  uint8_t n_classes,
                                  uint8_t feature_idx,
                                  const ht_criterion_vtable_t *crit,
                                  ht_float_t min_branch_frac,
                                  ht_split_candidate_t *out_best)
{
    if (out_best == NULL) {
        return;
    }
    ht_split_candidate_init(out_best);
    out_best->is_categorical = 0;

    if (obs == NULL || pre_dist == NULL || crit == NULL || crit->merit_of_split == NULL ||
        n_classes == 0 || n_classes > HT_MAX_CLASSES) {
        return;
    }

    ht_float_t splits[HT_GAUSSIAN_SPLITS];
    uint8_t n_splits = ht_gaussian_split_suggestions(obs, n_classes, splits);
    if (n_splits == 0) {
        return;
    }

    for (uint8_t i = 0; i < n_splits; i++) {
        ht_float_t lhs[HT_MAX_CLASSES];
        ht_float_t rhs[HT_MAX_CLASSES];
        ht_gaussian_binary_split_dists(obs, n_classes, splits[i], lhs, rhs);

        ht_float_t merit = crit->merit_of_split(pre_dist, lhs, rhs, n_classes, min_branch_frac);
        if (merit > out_best->merit) {
            out_best->merit = merit;
            out_best->feature_idx = feature_idx;
            out_best->split_val.threshold = splits[i];
            for (uint8_t c = 0; c < HT_MAX_CLASSES; c++) {
                out_best->lhs_dist[c] = lhs[c];
                out_best->rhs_dist[c] = rhs[c];
            }
        }
    }
}

void ht_evaluate_categorical_observer(const ht_categorical_obs_t *obs,
                                      const ht_float_t pre_dist[HT_MAX_CLASSES],
                                      uint8_t n_classes,
                                      uint8_t feature_idx,
                                      const ht_criterion_vtable_t *crit,
                                      ht_float_t min_branch_frac,
                                      ht_split_candidate_t *out_best)
{
    if (out_best == NULL) {
        return;
    }
    ht_split_candidate_init(out_best);
    out_best->is_categorical = 1;

    if (obs == NULL || pre_dist == NULL || crit == NULL || crit->merit_of_split == NULL ||
        n_classes == 0 || n_classes > HT_MAX_CLASSES) {
        return;
    }

    for (int32_t cat_v = 0; cat_v < (int32_t)HT_MAX_CATEGORIES; cat_v++) {
        if (!ht_categorical_is_observed(obs, cat_v)) {
            continue;
        }

        ht_float_t lhs[HT_MAX_CLASSES];
        ht_float_t rhs[HT_MAX_CLASSES];
        ht_categorical_binary_split_dists(obs, n_classes, cat_v, lhs, rhs);

        ht_float_t merit = crit->merit_of_split(pre_dist, lhs, rhs, n_classes, min_branch_frac);
        if (merit > out_best->merit) {
            out_best->merit = merit;
            out_best->feature_idx = feature_idx;
            out_best->split_val.cat_val = cat_v;
            for (uint8_t c = 0; c < HT_MAX_CLASSES; c++) {
                out_best->lhs_dist[c] = lhs[c];
                out_best->rhs_dist[c] = rhs[c];
            }
        }
    }
}

void ht_evaluate_hist_observer(const ht_hist_obs_t *obs,
                               const ht_float_t pre_dist[HT_MAX_CLASSES],
                               uint8_t n_classes,
                               uint8_t feature_idx,
                               const ht_criterion_vtable_t *crit,
                               ht_float_t min_branch_frac,
                               ht_split_candidate_t *out_best)
{
    if (out_best == NULL) {
        return;
    }
    ht_split_candidate_init(out_best);
    out_best->is_categorical = 0;

    if (obs == NULL || pre_dist == NULL || crit == NULL || crit->merit_of_split == NULL ||
        n_classes == 0 || n_classes > HT_MAX_CLASSES) {
        return;
    }

    ht_float_t splits[HT_GAUSSIAN_SPLITS];
    uint8_t n_splits = ht_hist_split_suggestions(obs, n_classes, splits);
    if (n_splits == 0) {
        return;
    }

    for (uint8_t i = 0; i < n_splits; i++) {
        ht_float_t lhs[HT_MAX_CLASSES];
        ht_float_t rhs[HT_MAX_CLASSES];
        ht_hist_binary_split_dists(obs, n_classes, splits[i], lhs, rhs);

        ht_float_t merit = crit->merit_of_split(pre_dist, lhs, rhs, n_classes, min_branch_frac);
        if (merit > out_best->merit) {
            out_best->merit = merit;
            out_best->feature_idx = feature_idx;
            out_best->split_val.threshold = splits[i];
            for (uint8_t c = 0; c < HT_MAX_CLASSES; c++) {
                out_best->lhs_dist[c] = lhs[c];
                out_best->rhs_dist[c] = rhs[c];
            }
        }
    }
}

void ht_find_best_split_candidates_ext(const ht_leaf_stats_t *stats,
                                       const ht_float_t pre_dist[HT_MAX_CLASSES],
                                       const ht_feature_type_t feature_types[HT_MAX_FEATURES],
                                       uint8_t n_features,
                                       uint8_t n_classes,
                                       const ht_criterion_vtable_t *crit,
                                       ht_float_t min_branch_frac,
                                       bool merit_preprune,
                                       ht_numeric_obs_type_t obs_type,
                                       ht_split_candidate_t *out_best,
                                       ht_split_candidate_t *out_second_best)
{
    if (out_best != NULL) {
        ht_split_candidate_init(out_best);
    }
    if (out_second_best != NULL) {
        ht_split_candidate_init(out_second_best);
    }

    if (out_best == NULL || out_second_best == NULL || stats == NULL ||
        pre_dist == NULL || feature_types == NULL || crit == NULL ||
        n_features == 0 || n_features > HT_MAX_FEATURES ||
        n_classes == 0 || n_classes > HT_MAX_CLASSES) {
        return;
    }

    /* Stack array for all evaluated candidates (at most HT_MAX_FEATURES + 1 null split) */
    ht_split_candidate_t suggestions[HT_MAX_FEATURES + 1];
    uint8_t count = 0;

    if (merit_preprune) {
        ht_split_candidate_init(&suggestions[count]);
        suggestions[count].merit = -HT_INFINITY;
        suggestions[count].feature_idx = HT_NO_FEATURE;
        count++;
    }

    for (uint8_t f = 0; f < n_features; f++) {
        ht_split_candidate_t cand;
        if (feature_types[f] == HT_FEAT_NUMERIC) {
            if (obs_type == HT_OBS_HISTOGRAM) {
                ht_evaluate_hist_observer(&stats->observers[f].hist,
                                          pre_dist, n_classes, f,
                                          crit, min_branch_frac, &cand);
            } else {
                ht_evaluate_numeric_observer(stats->observers[f].num,
                                             pre_dist, n_classes, f,
                                             crit, min_branch_frac, &cand);
            }
        } else {
            ht_evaluate_categorical_observer(&stats->observers[f].cat,
                                             pre_dist, n_classes, f,
                                             crit, min_branch_frac, &cand);
        }
        suggestions[count++] = cand;
    }

    if (count == 0) {
        return;
    }

    /* Stable insertion sort matching River's Timsort on suggestions:
     * strictly key.merit < suggestions[j].merit preserves insertion order on ties. */
    for (uint8_t i = 1; i < count; i++) {
        ht_split_candidate_t key = suggestions[i];
        int j = (int)i - 1;
        while (j >= 0 && key.merit < suggestions[j].merit) {
            suggestions[j + 1] = suggestions[j];
            j--;
        }
        suggestions[j + 1] = key;
    }

    *out_best = suggestions[count - 1];
    if (count >= 2) {
        *out_second_best = suggestions[count - 2];
    }
}

void ht_find_best_split_candidates(const ht_leaf_stats_t *stats,
                                   const ht_float_t pre_dist[HT_MAX_CLASSES],
                                   const ht_feature_type_t feature_types[HT_MAX_FEATURES],
                                   uint8_t n_features,
                                   uint8_t n_classes,
                                   const ht_criterion_vtable_t *crit,
                                   ht_float_t min_branch_frac,
                                   bool merit_preprune,
                                   ht_split_candidate_t *out_best,
                                   ht_split_candidate_t *out_second_best)
{
    ht_find_best_split_candidates_ext(stats, pre_dist, feature_types, n_features, n_classes,
                                      crit, min_branch_frac, merit_preprune,
                                      HT_OBS_GAUSSIAN, out_best, out_second_best);
}

bool ht_should_split(const ht_split_candidate_t *best,
                     const ht_split_candidate_t *second_best,
                     ht_float_t hoeffding_bound,
                     ht_float_t tau)
{
    if (best == NULL || best->merit == -HT_INFINITY || best->feature_idx == HT_NO_FEATURE) {
        return false;
    }

    if (second_best == NULL || second_best->merit == -HT_INFINITY) {
        return true;
    }

    ht_float_t diff = HT_SUB(best->merit, second_best->merit);
    if (diff > hoeffding_bound || hoeffding_bound < tau) {
        return true;
    }

    return false;
}

/* ========================================================================= */
/* --- Regression Variance Reduction Split Criterion Implementation ------- */
/* ========================================================================= */

#ifdef HT_USE_FIXED_POINT
static ht_float_t ht_compute_vr_internal(const ht_var_stat_t *pre_var,
                                         const ht_var_stat_t *lhs_var,
                                         const ht_var_stat_t *rhs_var,
                                         ht_float_t min_samples_split,
                                         ht_float_t min_branch_frac,
                                         bool scale_large_var)
{
    if (pre_var == NULL || lhs_var == NULL || rhs_var == NULL) {
        return -HT_INFINITY;
    }

    ht_float_t n_pre = pre_var->count;
    ht_float_t n_l = lhs_var->count;
    ht_float_t n_r = rhs_var->count;

    if (n_pre <= HT_CONST(2.0) || n_l < min_samples_split || n_r < min_samples_split) {
        return -HT_INFINITY;
    }

    ht_float_t frac_l = HT_DIV(n_l, n_pre);
    ht_float_t frac_r = HT_DIV(n_r, n_pre);
    if (frac_l < min_branch_frac || frac_r < min_branch_frac) {
        return -HT_INFINITY;
    }

    int64_t var_pre_q8 = ht_var_value_q8(pre_var);
    int64_t var_l_q8 = ht_var_value_q8(lhs_var);
    int64_t var_r_q8 = ht_var_value_q8(rhs_var);

    int64_t post_q8 = (((int64_t)n_l * var_l_q8) + ((int64_t)n_r * var_r_q8)) / (int64_t)n_pre;
    int64_t vr_q8 = var_pre_q8 - post_q8;

    const unsigned int lshift = (unsigned int)(HT_FIXED_SHIFT - HT_FIXED_M2_SHIFT);
    if (scale_large_var && var_pre_q8 > (16384LL << HT_FIXED_M2_SHIFT)) {
        /* Scale consistently across all splits of this leaf so merit never saturates Q16.16
         * while preserving exact ordering and exact m2/m1 ratio for ht_should_split_reg. */
        int64_t scaled_m2 = (vr_q8 * (16384LL << HT_FIXED_M2_SHIFT)) / var_pre_q8;
        int64_t scaled_q16 = (scaled_m2 >= 0)
            ? (int64_t)((uint64_t)scaled_m2 << lshift)
            : -(int64_t)((uint64_t)(-scaled_m2) << lshift);
        return ht_fixed_saturate64(scaled_q16);
    }

    int64_t vr_q16 = (vr_q8 >= 0)
        ? (int64_t)((uint64_t)vr_q8 << lshift)
        : -(int64_t)((uint64_t)(-vr_q8) << lshift);
    return ht_fixed_saturate64(vr_q16);
}
#endif

ht_float_t ht_compute_variance_reduction(const ht_var_stat_t *pre_var,
                                         const ht_var_stat_t *lhs_var,
                                         const ht_var_stat_t *rhs_var,
                                         ht_float_t min_samples_split,
                                         ht_float_t min_branch_frac)
{
#ifdef HT_USE_FIXED_POINT
    return ht_compute_vr_internal(pre_var, lhs_var, rhs_var, min_samples_split, min_branch_frac, false);
#else
    if (pre_var == NULL || lhs_var == NULL || rhs_var == NULL) {
        return -HT_INFINITY;
    }

    ht_float_t n_pre = pre_var->count;
    ht_float_t n_l = lhs_var->count;
    ht_float_t n_r = rhs_var->count;

    if (n_pre <= HT_CONST(2.0) || n_l < min_samples_split || n_r < min_samples_split) {
        return -HT_INFINITY;
    }

    if (HT_DIV(n_l, n_pre) < min_branch_frac || HT_DIV(n_r, n_pre) < min_branch_frac) {
        return -HT_INFINITY;
    }

    ht_float_t var_pre = ht_var_value(pre_var);
    ht_float_t var_l = ht_var_value(lhs_var);
    ht_float_t var_r = ht_var_value(rhs_var);

    ht_float_t vr = HT_SUB(var_pre,
                           HT_ADD(HT_MUL(HT_DIV(n_l, n_pre), var_l),
                                  HT_MUL(HT_DIV(n_r, n_pre), var_r)));
    return vr;
#endif
}

void ht_evaluate_reg_numeric_observer(const ht_reg_numeric_obs_t *obs,
                                      const ht_var_stat_t *pre_var,
                                      uint8_t feature_idx,
                                      ht_float_t min_samples_split,
                                      ht_float_t min_branch_frac,
                                      ht_split_candidate_t *out_best)
{
    if (out_best == NULL) {
        return;
    }
    ht_split_candidate_init(out_best);
    out_best->is_categorical = 0;

    if (obs == NULL || pre_var == NULL) {
        return;
    }

    /* Count active slots in obs->slots (slots are strictly packed 0..m-1) */
    uint8_t m = 0;
    while (m < HT_REG_NUM_SLOTS && obs->slots[m].y_stats.count > HT_CONST(0.0)) {
        m++;
    }

    if (m < 2) {
        return;
    }

#ifdef HT_USE_FIXED_POINT
    /* Precompute suffix combinations for scale-invariant rhs_var under Block-Rescaling */
    ht_var_stat_t suffix[HT_REG_NUM_SLOTS];
    suffix[m - 1] = obs->slots[m - 1].y_stats;
    for (int32_t j = (int32_t)m - 2; j >= 0; --j) {
        suffix[j] = ht_var_combine(&obs->slots[j].y_stats, &suffix[j + 1]);
    }
#endif

    ht_var_stat_t lhs_var;
    ht_var_init(&lhs_var);

    for (uint8_t k = 0; k < m - 1; k++) {
        if (k == 0) {
            lhs_var = obs->slots[0].y_stats;
        } else {
            lhs_var = ht_var_combine(&lhs_var, &obs->slots[k].y_stats);
        }

        /* Skip duplicate adjacent x_means */
        if (obs->slots[k + 1].x_mean <= obs->slots[k].x_mean) {
            continue;
        }

        ht_float_t threshold = HT_MUL(HT_ADD(obs->slots[k].x_mean, obs->slots[k + 1].x_mean), HT_CONST(0.5));

#ifdef HT_USE_FIXED_POINT
        ht_var_stat_t rhs_var = suffix[k + 1];
        const ht_var_stat_t *eff_pre = &suffix[0];
        ht_float_t merit = ht_compute_vr_internal(eff_pre, &lhs_var, &rhs_var,
                                                  min_samples_split, min_branch_frac, true);
#else
        ht_var_stat_t rhs_var = ht_var_subtract(pre_var, &lhs_var);
        ht_float_t merit = ht_compute_variance_reduction(pre_var, &lhs_var, &rhs_var,
                                                         min_samples_split, min_branch_frac);
#endif

        if (merit > out_best->merit) {
            out_best->merit = merit;
            out_best->feature_idx = feature_idx;
            out_best->split_val.threshold = threshold;
            out_best->lhs_var = lhs_var;
            out_best->rhs_var = rhs_var;
            out_best->is_categorical = 0;
        }
    }
}

void ht_evaluate_reg_categorical_observer(const ht_reg_categorical_obs_t *obs,
                                          const ht_var_stat_t *pre_var,
                                          uint8_t feature_idx,
                                          ht_float_t min_samples_split,
                                          ht_float_t min_branch_frac,
                                          ht_split_candidate_t *out_best)
{
    if (out_best == NULL) {
        return;
    }
    ht_split_candidate_init(out_best);
    out_best->is_categorical = 1;

    if (obs == NULL || pre_var == NULL) {
        return;
    }

    for (int32_t cat_v = 0; cat_v < (int32_t)HT_MAX_CATEGORIES; cat_v++) {
        if (obs->cat_stats[cat_v].count <= HT_CONST(0.0)) {
            continue;
        }

        const ht_var_stat_t *lhs_var = &obs->cat_stats[cat_v];
#ifdef HT_USE_FIXED_POINT
        ht_var_stat_t rhs_var;
        ht_var_init(&rhs_var);
        for (int32_t u = 0; u < (int32_t)HT_MAX_CATEGORIES; u++) {
            if (u != cat_v && obs->cat_stats[u].count > 0) {
                rhs_var = ht_var_combine(&rhs_var, &obs->cat_stats[u]);
            }
        }
        ht_var_stat_t eff_pre = ht_var_combine(lhs_var, &rhs_var);
        ht_float_t merit = ht_compute_vr_internal(&eff_pre, lhs_var, &rhs_var,
                                                  min_samples_split, min_branch_frac, true);
#else
        ht_var_stat_t rhs_var = ht_var_subtract(pre_var, lhs_var);
        ht_float_t merit = ht_compute_variance_reduction(pre_var, lhs_var, &rhs_var,
                                                         min_samples_split, min_branch_frac);
#endif

        if (merit > out_best->merit) {
            out_best->merit = merit;
            out_best->feature_idx = feature_idx;
            out_best->split_val.cat_val = cat_v;
            out_best->lhs_var = *lhs_var;
            out_best->rhs_var = rhs_var;
            out_best->is_categorical = 1;
        }
    }
}

void ht_find_best_reg_split_candidates(const ht_leaf_stats_t *stats,
                                       const ht_var_stat_t *pre_var,
                                       const ht_feature_type_t feature_types[HT_MAX_FEATURES],
                                       uint8_t n_features,
                                       ht_float_t min_samples_split,
                                       ht_float_t min_branch_frac,
                                       bool merit_preprune,
                                       ht_split_candidate_t *out_best,
                                       ht_split_candidate_t *out_second_best)
{
    if (out_best != NULL) {
        ht_split_candidate_init(out_best);
    }
    if (out_second_best != NULL) {
        ht_split_candidate_init(out_second_best);
    }

    if (out_best == NULL || out_second_best == NULL || stats == NULL ||
        pre_var == NULL || feature_types == NULL ||
        n_features == 0 || n_features > HT_MAX_FEATURES) {
        return;
    }

    /* Stack array for all evaluated candidates (at most HT_MAX_FEATURES + 1 null split) */
    ht_split_candidate_t suggestions[HT_MAX_FEATURES + 1];
    uint8_t count = 0;

    if (merit_preprune) {
        ht_split_candidate_init(&suggestions[count]);
        suggestions[count].merit = -HT_INFINITY;
        suggestions[count].feature_idx = HT_NO_FEATURE;
        count++;
    }

    for (uint8_t f = 0; f < n_features; f++) {
        ht_split_candidate_t cand;
        if (feature_types[f] == HT_FEAT_NUMERIC) {
            ht_evaluate_reg_numeric_observer(&stats->observers[f].reg_num,
                                             pre_var, f,
                                             min_samples_split, min_branch_frac,
                                             &cand);
        } else {
            ht_evaluate_reg_categorical_observer(&stats->observers[f].reg_cat,
                                                 pre_var, f,
                                                 min_samples_split, min_branch_frac,
                                                 &cand);
        }
        suggestions[count++] = cand;
    }

    if (count == 0) {
        return;
    }

    /* Stable insertion sort matching River's Timsort on suggestions:
     * strictly key.merit < suggestions[j].merit preserves insertion order on ties. */
    for (uint8_t i = 1; i < count; i++) {
        ht_split_candidate_t key = suggestions[i];
        int j = (int)i - 1;
        while (j >= 0 && key.merit < suggestions[j].merit) {
            suggestions[j + 1] = suggestions[j];
            j--;
        }
        suggestions[j + 1] = key;
    }

    *out_best = suggestions[count - 1];
    if (count >= 2) {
        *out_second_best = suggestions[count - 2];
    }
}

bool ht_should_split_reg(const ht_split_candidate_t *best,
                         const ht_split_candidate_t *second_best,
                         ht_float_t hoeffding_bound,
                         ht_float_t tau)
{
    if (best == NULL || best->merit <= HT_CONST(0.0) || best->feature_idx == HT_NO_FEATURE) {
        return false;
    }

    if (second_best == NULL || second_best->feature_idx == HT_NO_FEATURE ||
        second_best->merit == -HT_INFINITY || second_best->merit <= HT_CONST(0.0)) {
        return true;
    }

    ht_float_t ratio = HT_DIV(second_best->merit, best->merit);
    if (ratio < HT_SUB(HT_CONST(1.0), hoeffding_bound) || hoeffding_bound < tau) {
        return true;
    }

    return false;
}
