/**
 * @file ht_observer.c
 * @brief Implementation of Attribute Observers for embedded VFDT / HAT.
 */

#include "ht_observer.h"
#include <string.h>

#define HT_TWO_PI ((ht_float_t)(2.0 * 3.14159265358979323846))

/* ========================================================================= */
/* --- Gaussian Numeric Observer Implementation ---------------------------- */
/* ========================================================================= */

void ht_gaussian_init(ht_gaussian_obs_t *obs) {
    if (obs) {
        memset(obs, 0, sizeof(*obs));
    }
}

void ht_gaussian_update(ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                        uint8_t class_idx,
                        ht_float_t val,
                        ht_float_t weight)
{
    if (!obs || class_idx >= HT_MAX_CLASSES || weight <= (ht_float_t)0.0 ||
        HT_ISNAN(val) || !HT_ISFINITE(val)) {
        return;
    }

    ht_gaussian_obs_t *o = &obs[class_idx];

    if (o->count <= (ht_float_t)0.0) {
        o->count = weight;
        o->mean = val;
        o->m2 = (ht_float_t)0.0;
        o->min_val = val;
        o->max_val = val;
    } else {
        ht_float_t old_count = o->count;
        ht_float_t new_count = old_count + weight;
        ht_float_t delta = val - o->mean;

        o->mean += (weight / new_count) * delta;
        ht_float_t delta2 = val - o->mean;
        o->m2 += weight * delta * delta2;
        o->count = new_count;

        if (val < o->min_val) {
            o->min_val = val;
        }
        if (val > o->max_val) {
            o->max_val = val;
        }
    }
}

ht_float_t ht_gaussian_variance(const ht_gaussian_obs_t *obs) {
    if (!obs || obs->count <= (ht_float_t)1.0 || obs->m2 <= (ht_float_t)0.0) {
        return (ht_float_t)0.0;
    }

    ht_float_t var = obs->m2 / (obs->count - (ht_float_t)1.0);
    return (var > (ht_float_t)0.0) ? var : (ht_float_t)0.0;
}

ht_float_t ht_gaussian_stddev(const ht_gaussian_obs_t *obs) {
    ht_float_t var = ht_gaussian_variance(obs);
    return (var > (ht_float_t)0.0) ? HT_SQRT(var) : (ht_float_t)0.0;
}

ht_float_t ht_normal_cdf(ht_float_t x, ht_float_t mean, ht_float_t stddev) {
    if (HT_ISNAN(x) || HT_ISNAN(mean)) {
        return (ht_float_t)0.0 / (ht_float_t)0.0;
    }

    if (stddev <= (ht_float_t)0.0 || HT_ISNAN(stddev)) {
        if (x < mean) {
            return (ht_float_t)0.0;
        }
        if (x > mean) {
            return (ht_float_t)1.0;
        }
        return (ht_float_t)0.5;
    }

    ht_float_t z = (x - mean) / (stddev * HT_SQRT2);
    ht_float_t cdf = (ht_float_t)0.5 * ((ht_float_t)1.0 + HT_ERF(z));

    if (cdf < (ht_float_t)0.0) {
        cdf = (ht_float_t)0.0;
    } else if (cdf > (ht_float_t)1.0) {
        cdf = (ht_float_t)1.0;
    }

    return cdf;
}

ht_float_t ht_gaussian_cond_proba(const ht_gaussian_obs_t *obs, ht_float_t val) {
    if (!obs || obs->count < (ht_float_t)2.0 || HT_ISNAN(val) || !HT_ISFINITE(val)) {
        return (ht_float_t)0.0;
    }

    ht_float_t s2 = ht_gaussian_variance(obs);
    if (s2 <= (ht_float_t)0.0 || HT_ISNAN(s2)) {
        return (ht_float_t)0.0;
    }

    ht_float_t diff = val - obs->mean;
    ht_float_t stddev = HT_SQRT(s2);
    ht_float_t inv_norm = (ht_float_t)1.0 / (stddev * HT_SQRT(HT_TWO_PI));
    ht_float_t exponent = -(diff * diff) / ((ht_float_t)2.0 * s2);

    return inv_norm * HT_EXP(exponent);
}

ht_float_t ht_gaussian_cond_log_proba(const ht_gaussian_obs_t *obs, ht_float_t val) {
    if (!obs || obs->count < (ht_float_t)2.0 || HT_ISNAN(val) || !HT_ISFINITE(val)) {
        return -HT_INFINITY;
    }

    ht_float_t s2 = ht_gaussian_variance(obs);
    if (s2 <= (ht_float_t)0.0 || HT_ISNAN(s2)) {
        return -HT_INFINITY;
    }

    ht_float_t diff = val - obs->mean;
    ht_float_t log_two_pi_s2 = HT_LOG(HT_TWO_PI) + HT_LOG(s2);
    ht_float_t exponent_term = (diff * diff) / ((ht_float_t)2.0 * s2);

    return (ht_float_t)-0.5 * log_two_pi_s2 - exponent_term;
}

uint8_t ht_gaussian_split_suggestions(const ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                                      uint8_t n_classes,
                                      ht_float_t splits_out[HT_GAUSSIAN_SPLITS])
{
    if (!obs || !splits_out || n_classes == 0) {
        return 0;
    }

    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;
    ht_float_t min_val = HT_INFINITY;
    ht_float_t max_val = -HT_INFINITY;
    bool found = false;

    for (uint8_t c = 0; c < max_c; ++c) {
        if (obs[c].count > (ht_float_t)0.0) {
            if (obs[c].min_val < min_val) {
                min_val = obs[c].min_val;
            }
            if (obs[c].max_val > max_val) {
                max_val = obs[c].max_val;
            }
            found = true;
        }
    }

    if (!found || min_val >= max_val) {
        return 0;
    }

    ht_float_t bin_size = (max_val - min_val) / (ht_float_t)(HT_GAUSSIAN_SPLITS + 1);
    uint8_t count = 0;

    for (uint8_t i = 0; i < HT_GAUSSIAN_SPLITS; ++i) {
        ht_float_t split_val = min_val + bin_size * (ht_float_t)(i + 1);
        if (split_val > min_val && split_val < max_val) {
            splits_out[count++] = split_val;
        }
    }

    return count;
}

void ht_gaussian_binary_split_dists(const ht_gaussian_obs_t obs[HT_MAX_CLASSES],
                                    uint8_t n_classes,
                                    ht_float_t split_val,
                                    ht_float_t lhs_dist[HT_MAX_CLASSES],
                                    ht_float_t rhs_dist[HT_MAX_CLASSES])
{
    if (lhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            lhs_dist[c] = (ht_float_t)0.0;
        }
    }
    if (rhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            rhs_dist[c] = (ht_float_t)0.0;
        }
    }

    if (!obs || !lhs_dist || !rhs_dist) {
        return;
    }

    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;

    for (uint8_t c = 0; c < max_c; ++c) {
        if (obs[c].count <= (ht_float_t)0.0) {
            continue;
        }

        if (split_val < obs[c].min_val) {
            rhs_dist[c] = obs[c].count;
        } else if (split_val >= obs[c].max_val) {
            lhs_dist[c] = obs[c].count;
        } else {
            ht_float_t stddev = ht_gaussian_stddev(&obs[c]);
            ht_float_t cdf = ht_normal_cdf(split_val, obs[c].mean, stddev);
            ht_float_t left_w = cdf * obs[c].count;
            lhs_dist[c] = left_w;
            ht_float_t right_w = obs[c].count - left_w;
            rhs_dist[c] = (right_w > (ht_float_t)0.0) ? right_w : (ht_float_t)0.0;
        }
    }
}

/* ========================================================================= */
/* --- Categorical Nominal Observer Implementation ------------------------- */
/* ========================================================================= */

void ht_categorical_init(ht_categorical_obs_t *obs) {
    if (obs) {
        memset(obs->counts, 0, sizeof(obs->counts));
    }
}

void ht_categorical_update(ht_categorical_obs_t *obs,
                           uint8_t class_idx,
                           int32_t cat_val,
                           ht_float_t weight)
{
    if (!obs || class_idx >= HT_MAX_CLASSES || cat_val < 0 ||
        cat_val >= (int32_t)HT_MAX_CATEGORIES || weight <= (ht_float_t)0.0) {
        return;
    }

    obs->counts[class_idx][cat_val] += weight;
}

ht_float_t ht_categorical_cond_proba(const ht_categorical_obs_t *obs,
                                     uint8_t class_idx,
                                     int32_t cat_val)
{
    if (!obs || class_idx >= HT_MAX_CLASSES || cat_val < 0 ||
        cat_val >= (int32_t)HT_MAX_CATEGORIES) {
        return (ht_float_t)0.0;
    }

    ht_float_t total = (ht_float_t)0.0;
    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
        total += obs->counts[class_idx][v];
    }

    if (total <= (ht_float_t)0.0) {
        return (ht_float_t)0.0;
    }

    return obs->counts[class_idx][cat_val] / total;
}

bool ht_categorical_is_observed(const ht_categorical_obs_t *obs, int32_t cat_val) {
    if (!obs || cat_val < 0 || cat_val >= (int32_t)HT_MAX_CATEGORIES) {
        return false;
    }

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        if (obs->counts[c][cat_val] > (ht_float_t)0.0) {
            return true;
        }
    }

    return false;
}

void ht_categorical_binary_split_dists(const ht_categorical_obs_t *obs,
                                       uint8_t n_classes,
                                       int32_t cat_val,
                                       ht_float_t lhs_dist[HT_MAX_CLASSES],
                                       ht_float_t rhs_dist[HT_MAX_CLASSES])
{
    if (lhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            lhs_dist[c] = (ht_float_t)0.0;
        }
    }
    if (rhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            rhs_dist[c] = (ht_float_t)0.0;
        }
    }

    if (!obs || !lhs_dist || !rhs_dist) {
        return;
    }

    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;

    for (uint8_t c = 0; c < max_c; ++c) {
        ht_float_t eq_w = (ht_float_t)0.0;
        ht_float_t not_eq_w = (ht_float_t)0.0;

        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            ht_float_t w = obs->counts[c][v];
            if (cat_val >= 0 && v == (uint8_t)cat_val) {
                eq_w += w;
            } else {
                not_eq_w += w;
            }
        }

        lhs_dist[c] = eq_w;
        rhs_dist[c] = not_eq_w;
    }
}

/* ========================================================================= */
/* --- Candidate Helper Implementation ------------------------------------- */
/* ========================================================================= */

void ht_split_candidate_init(ht_split_candidate_t *cand) {
    if (!cand) {
        return;
    }

    cand->merit = -HT_INFINITY;
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        cand->lhs_dist[c] = (ht_float_t)0.0;
        cand->rhs_dist[c] = (ht_float_t)0.0;
    }
    cand->split_val.threshold = (ht_float_t)0.0;
    cand->feature_idx = HT_NO_FEATURE;
    cand->is_categorical = 0;
}

/* ========================================================================= */
/* --- Incremental Variance and Regression Observer Implementation --------- */
/* ========================================================================= */

void ht_var_init(ht_var_stat_t *var) {
    if (var) {
        var->count = (ht_float_t)0.0;
        var->mean = (ht_float_t)0.0;
        var->m2 = (ht_float_t)0.0;
    }
}

void ht_var_update(ht_var_stat_t *var, ht_float_t y, ht_float_t weight) {
    if (!var || weight <= (ht_float_t)0.0 || HT_ISNAN(y) || !HT_ISFINITE(y)) {
        return;
    }

    if (var->count <= (ht_float_t)0.0) {
        var->count = weight;
        var->mean = y;
        var->m2 = (ht_float_t)0.0;
    } else {
        ht_float_t old_count = var->count;
        ht_float_t new_count = old_count + weight;
        ht_float_t delta = y - var->mean;

        var->mean += (weight / new_count) * delta;
        ht_float_t delta2 = y - var->mean;
        var->m2 += weight * delta * delta2;
        var->count = new_count;

        if (var->m2 < (ht_float_t)0.0) {
            var->m2 = (ht_float_t)0.0;
        }
    }
}

ht_float_t ht_var_value(const ht_var_stat_t *var) {
    if (!var || var->count <= (ht_float_t)1.0 || var->m2 <= (ht_float_t)0.0) {
        return (ht_float_t)0.0;
    }
    ht_float_t v = var->m2 / (var->count - (ht_float_t)1.0);
    return (v > (ht_float_t)0.0) ? v : (ht_float_t)0.0;
}

ht_var_stat_t ht_var_combine(const ht_var_stat_t *a, const ht_var_stat_t *b) {
    ht_var_stat_t res;
    res.count = (ht_float_t)0.0;
    res.mean = (ht_float_t)0.0;
    res.m2 = (ht_float_t)0.0;

    bool a_valid = (a && a->count > (ht_float_t)0.0);
    bool b_valid = (b && b->count > (ht_float_t)0.0);

    if (!a_valid && !b_valid) {
        return res;
    }
    if (!a_valid) {
        return *b;
    }
    if (!b_valid) {
        return *a;
    }

    res.count = a->count + b->count;
    ht_float_t delta = b->mean - a->mean;
    res.mean = a->mean + (b->count / res.count) * delta;
    res.m2 = a->m2 + b->m2 + (delta * delta) * ((a->count * b->count) / res.count);

    if (res.m2 < (ht_float_t)0.0) {
        res.m2 = (ht_float_t)0.0;
    }

    return res;
}

ht_var_stat_t ht_var_subtract(const ht_var_stat_t *total, const ht_var_stat_t *part) {
    ht_var_stat_t res;
    res.count = (ht_float_t)0.0;
    res.mean = (ht_float_t)0.0;
    res.m2 = (ht_float_t)0.0;

    if (!total || total->count <= (ht_float_t)0.0) {
        return res;
    }
    if (!part || part->count <= (ht_float_t)0.0) {
        return *total;
    }

    if (part->count >= total->count) {
        return res;
    }

    res.count = total->count - part->count;
    if (res.count <= (ht_float_t)0.0) {
        return res;
    }

    res.mean = (total->count * total->mean - part->count * part->mean) / res.count;
    ht_float_t delta = res.mean - part->mean;
    ht_float_t diff_m2 = total->m2 - part->m2 - (delta * delta) * ((res.count * part->count) / total->count);

    res.m2 = (diff_m2 > (ht_float_t)0.0) ? diff_m2 : (ht_float_t)0.0;

    return res;
}

void ht_reg_numeric_init(ht_reg_numeric_obs_t *obs) {
    if (obs) {
        memset(obs, 0, sizeof(*obs));
    }
}

void ht_reg_numeric_update(ht_reg_numeric_obs_t *obs,
                           ht_float_t x_val,
                           ht_float_t y_val,
                           ht_float_t weight)
{
    if (!obs || weight <= (ht_float_t)0.0 ||
        HT_ISNAN(x_val) || !HT_ISFINITE(x_val) ||
        HT_ISNAN(y_val) || !HT_ISFINITE(y_val)) {
        return;
    }

    /* Count active slots in obs->slots (slots are strictly packed 0..m-1) */
    uint8_t m = 0;
    while (m < HT_REG_NUM_SLOTS && obs->slots[m].y_stats.count > (ht_float_t)0.0) {
        m++;
    }

    if (m < HT_REG_NUM_SLOTS) {
        /* Room available: insert new slot in ascending order of x_mean */
        uint8_t pos = 0;
        while (pos < m && obs->slots[pos].x_mean <= x_val) {
            pos++;
        }
        for (uint8_t i = m; i > pos; --i) {
            obs->slots[i] = obs->slots[i - 1];
        }
        obs->slots[pos].x_mean = x_val;
        obs->slots[pos].y_stats.count = weight;
        obs->slots[pos].y_stats.mean = y_val;
        obs->slots[pos].y_stats.m2 = (ht_float_t)0.0;
        return;
    }

    /* All HT_REG_NUM_SLOTS slots are occupied.
     * Find adjacent pair with minimum distance between their x_means. */
    uint8_t min_idx = 0;
    ht_float_t min_dist = obs->slots[1].x_mean - obs->slots[0].x_mean;
    for (uint8_t i = 1; i < HT_REG_NUM_SLOTS - 1; ++i) {
        ht_float_t dist = obs->slots[i + 1].x_mean - obs->slots[i].x_mean;
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }

    /* Fuse slots min_idx and min_idx + 1 using Chan's parallel formulas */
    ht_reg_num_slot_t *s0 = &obs->slots[min_idx];
    ht_reg_num_slot_t *s1 = &obs->slots[min_idx + 1];

    ht_float_t fused_count = s0->y_stats.count + s1->y_stats.count;
    ht_float_t fused_x = (s0->y_stats.count * s0->x_mean + s1->y_stats.count * s1->x_mean) / fused_count;
    ht_var_stat_t fused_y = ht_var_combine(&s0->y_stats, &s1->y_stats);

    s0->x_mean = fused_x;
    s0->y_stats = fused_y;

    /* Shift remaining slots left by 1 position */
    for (uint8_t i = min_idx + 1; i < HT_REG_NUM_SLOTS - 1; ++i) {
        obs->slots[i] = obs->slots[i + 1];
    }
    uint8_t active_count = (uint8_t)(HT_REG_NUM_SLOTS - 1U);

    /* Insert new sample (x_val, y_val, weight) in sorted order */
    uint8_t pos = 0;
    while (pos < active_count && obs->slots[pos].x_mean <= x_val) {
        pos++;
    }
    for (uint8_t i = active_count; i > pos; --i) {
        obs->slots[i] = obs->slots[i - 1];
    }
    obs->slots[pos].x_mean = x_val;
    obs->slots[pos].y_stats.count = weight;
    obs->slots[pos].y_stats.mean = y_val;
    obs->slots[pos].y_stats.m2 = (ht_float_t)0.0;
}

void ht_reg_categorical_init(ht_reg_categorical_obs_t *obs) {
    if (obs) {
        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            ht_var_init(&obs->cat_stats[v]);
        }
    }
}

void ht_reg_categorical_update(ht_reg_categorical_obs_t *obs,
                               int32_t cat_val,
                               ht_float_t y_val,
                               ht_float_t weight)
{
    if (!obs || cat_val < 0 || cat_val >= (int32_t)HT_MAX_CATEGORIES ||
        weight <= (ht_float_t)0.0 || HT_ISNAN(y_val) || !HT_ISFINITE(y_val)) {
        return;
    }

    ht_var_update(&obs->cat_stats[cat_val], y_val, weight);
}
