/**
 * @file ht_criterion.c
 * @brief Implementation of split criteria (Gini, InfoGain) and Hoeffding bound evaluation.
 */

#include "ht_criterion.h"

/* ========================================================================= */
/* --- Gini Impurity Criterion --------------------------------------------- */
/* ========================================================================= */

ht_float_t ht_compute_gini(const ht_float_t dist[HT_MAX_CLASSES],
                           uint8_t n_classes,
                           ht_float_t dist_sum)
{
    ht_float_t gini = (ht_float_t)1.0;
    if (dist_sum > (ht_float_t)0.0) {
        for (uint8_t c = 0; c < n_classes; c++) {
            if (dist[c] > (ht_float_t)0.0) {
                ht_float_t rel_freq = dist[c] / dist_sum;
                gini -= rel_freq * rel_freq;
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
    return (ht_float_t)1.0;
}

static ht_float_t ht_gini_merit_of_split(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                         const ht_float_t lhs_dist[HT_MAX_CLASSES],
                                         const ht_float_t rhs_dist[HT_MAX_CLASSES],
                                         uint8_t n_classes,
                                         ht_float_t min_branch_fraction)
{
    (void)pre_dist;
    ht_float_t w_l = (ht_float_t)0.0;
    ht_float_t w_r = (ht_float_t)0.0;

    for (uint8_t c = 0; c < n_classes; c++) {
        if (lhs_dist[c] > (ht_float_t)0.0) {
            w_l += lhs_dist[c];
        }
        if (rhs_dist[c] > (ht_float_t)0.0) {
            w_r += rhs_dist[c];
        }
    }

    ht_float_t total_weight = w_l + w_r;
    if (total_weight <= (ht_float_t)0.0) {
        return -HT_INFINITY;
    }

    if ((w_l / total_weight) <= min_branch_fraction ||
        (w_r / total_weight) <= min_branch_fraction) {
        return -HT_INFINITY;
    }

    ht_float_t gini = (w_l / total_weight) * ht_compute_gini(lhs_dist, n_classes, w_l) +
                      (w_r / total_weight) * ht_compute_gini(rhs_dist, n_classes, w_r);

    return (ht_float_t)1.0 - gini;
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
    ht_float_t entropy = (ht_float_t)0.0;
    ht_float_t dis_sums = (ht_float_t)0.0;

    for (uint8_t c = 0; c < n_classes; c++) {
        if (dist[c] > (ht_float_t)0.0) {
            entropy -= dist[c] * HT_LOG2(dist[c]);
            dis_sums += dist[c];
        }
    }

    if (dis_sums > (ht_float_t)0.0) {
        return (entropy + dis_sums * HT_LOG2(dis_sums)) / dis_sums;
    }

    return (ht_float_t)0.0;
}

static ht_float_t ht_info_gain_range_of_merit(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                              uint8_t n_classes)
{
    uint8_t num_observed = 0;
    for (uint8_t c = 0; c < n_classes; c++) {
        if (pre_dist[c] > (ht_float_t)0.0) {
            num_observed++;
        }
    }

    uint8_t k = (num_observed > 2) ? num_observed : 2;
    return HT_LOG2((ht_float_t)k);
}

static ht_float_t ht_info_gain_merit_of_split(const ht_float_t pre_dist[HT_MAX_CLASSES],
                                              const ht_float_t lhs_dist[HT_MAX_CLASSES],
                                              const ht_float_t rhs_dist[HT_MAX_CLASSES],
                                              uint8_t n_classes,
                                              ht_float_t min_branch_fraction)
{
    ht_float_t w_l = (ht_float_t)0.0;
    ht_float_t w_r = (ht_float_t)0.0;

    for (uint8_t c = 0; c < n_classes; c++) {
        if (lhs_dist[c] > (ht_float_t)0.0) {
            w_l += lhs_dist[c];
        }
        if (rhs_dist[c] > (ht_float_t)0.0) {
            w_r += rhs_dist[c];
        }
    }

    ht_float_t total_weight = w_l + w_r;
    if (total_weight <= (ht_float_t)0.0) {
        return -HT_INFINITY;
    }

    if ((w_l / total_weight) <= min_branch_fraction ||
        (w_r / total_weight) <= min_branch_fraction) {
        return -HT_INFINITY;
    }

    ht_float_t post_entropy = (w_l * ht_compute_entropy(lhs_dist, n_classes) +
                               w_r * ht_compute_entropy(rhs_dist, n_classes)) / total_weight;
    ht_float_t pre_entropy = ht_compute_entropy(pre_dist, n_classes);

    return pre_entropy - post_entropy;
}

static const ht_criterion_vtable_t s_info_gain_vtable = {
    .range_of_merit = ht_info_gain_range_of_merit,
    .merit_of_split = ht_info_gain_merit_of_split
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
        default:
            return NULL;
    }
}

ht_float_t ht_hoeffding_bound(ht_float_t range_val,
                              ht_float_t delta,
                              ht_float_t total_weight)
{
    if (total_weight <= (ht_float_t)0.0 || delta <= (ht_float_t)0.0) {
        return HT_INFINITY;
    }
    if (delta >= (ht_float_t)1.0) {
        return (ht_float_t)0.0;
    }
    return HT_SQRT((range_val * range_val * HT_LOG((ht_float_t)1.0 / delta)) /
                   ((ht_float_t)2.0 * total_weight));
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
            ht_evaluate_numeric_observer(stats->observers[f].num,
                                         pre_dist, n_classes, f,
                                         crit, min_branch_frac, &cand);
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

    ht_float_t diff = best->merit - second_best->merit;
    if (diff > hoeffding_bound || hoeffding_bound < tau) {
        return true;
    }

    return false;
}
