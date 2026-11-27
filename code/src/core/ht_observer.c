/**
 * @file ht_observer.c
 * @brief Implementation of Attribute Observers for embedded VFDT / HAT.
 *
 * Supports float32, double64, and Q16.16 zero-libm fixed-point arithmetic
 * (including Block-Rescaling and Q24.8 shifted variance m2 in ht_var_stat_t).
 */

#include "ht_observer.h"
#include <string.h>

#define HT_TWO_PI HT_CONST(6.28318530717958647692)
#define HT_SQRT_TWO_PI HT_CONST(2.50662827463100050242)

#ifdef HT_USE_FIXED_POINT
static void ht_gaussian_rescale_all(ht_gaussian_obs_t obs[HT_MAX_CLASSES]) {
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        if (obs[c].count > 0) {
            ht_float_t hc = ht_fixed_halve(obs[c].count);
            obs[c].count = (hc > HT_FIXED_ONE) ? hc : HT_FIXED_ONE;
            obs[c].m2 = ht_fixed_halve(obs[c].m2);
        }
    }
}

static void ht_categorical_rescale_all(ht_categorical_obs_t *obs) {
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            if (obs->counts[c][v] > 0) {
                ht_float_t h = ht_fixed_halve(obs->counts[c][v]);
                obs->counts[c][v] = (h > 0) ? h : 1;
            }
        }
    }
}

static void ht_hist_rescale_all(ht_hist_obs_t *obs, uint8_t n_bins) {
    for (uint8_t i = 0; i < n_bins; ++i) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            if (obs->bins[i].counts[c] > 0) {
                ht_float_t h = ht_fixed_halve(obs->bins[i].counts[c]);
                obs->bins[i].counts[c] = (h > 0) ? h : 1;
            }
        }
    }
}

static inline int64_t ht_q16_sq_to_q8(int64_t a, int64_t b) {
    const unsigned int rshift = (unsigned int)(32 - HT_FIXED_M2_SHIFT);
    const uint64_t round_bit = 1ULL << (rshift - 1U);
    int64_t prod = a * b;
    if (prod >= 0) {
        return (int64_t)(((uint64_t)prod + round_bit) >> rshift);
    }
    return -(int64_t)(((uint64_t)(-prod) + round_bit) >> rshift);
}

static inline int64_t ht_q8_mul_q16_to_q8(int64_t val_q8, int64_t w_q16) {
    if (val_q8 <= 0 || w_q16 <= 0) {
        return 0;
    }
    if (w_q16 == (int64_t)HT_FIXED_ONE) {
        return val_q8;
    }
    uint64_t uv = (uint64_t)val_q8;
    uint64_t uw = (uint64_t)w_q16;
    if (uv <= (1ULL << 32)) {
        return (int64_t)((uv * uw + 32768ULL) >> 16);
    }
    /* High dynamic range fallback: shift before multiply to avoid 64-bit overflow */
    return (int64_t)((uv >> 8) * (uw >> 8));
}
#endif

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
    if (!obs || class_idx >= HT_MAX_CLASSES || weight <= HT_CONST(0.0) ||
        HT_ISNAN(val) || !HT_ISFINITE(val)) {
        return;
    }

    ht_gaussian_obs_t *o = &obs[class_idx];

    if (o->count <= HT_CONST(0.0)) {
        o->count = weight;
        o->mean = val;
        o->m2 = HT_CONST(0.0);
        o->min_val = val;
        o->max_val = val;
    } else {
#ifdef HT_USE_FIXED_POINT
        int64_t old_count = (int64_t)o->count;
        int64_t w64 = (int64_t)weight;
        int64_t new_count = old_count + w64;
        int64_t delta = (int64_t)val - (int64_t)o->mean;
        int64_t new_mean = (int64_t)o->mean + ((delta * w64) / new_count);
        int64_t delta2 = (int64_t)val - new_mean;
        int64_t d1d2 = delta * delta2;
        int64_t d1d2_q16 = (d1d2 >= 0) ? (int64_t)(((uint64_t)d1d2 + 32768ULL) >> 16) : 0;
        int64_t m2_inc;
        if (w64 == (int64_t)HT_FIXED_ONE) {
            m2_inc = d1d2_q16;
        } else if (d1d2_q16 <= (1LL << 32)) {
            m2_inc = (int64_t)(((uint64_t)d1d2_q16 * (uint64_t)w64 + 32768ULL) >> 16);
        } else {
            m2_inc = (int64_t)(((uint64_t)d1d2_q16 >> 8) * ((uint64_t)w64 >> 8));
        }
        int64_t new_m2 = (int64_t)o->m2 + m2_inc;

        o->mean = ht_fixed_saturate64(new_mean);
        if ((new_count >= (int64_t)HT_FIXED_RESCALE_THRESHOLD ||
             new_m2 >= (int64_t)HT_FIXED_RESCALE_THRESHOLD) &&
            new_count >= (4LL * HT_FIXED_ONE)) {
            o->count = ht_fixed_saturate64(new_count);
            o->m2 = ht_fixed_saturate64(new_m2);
            ht_gaussian_rescale_all(obs);
            /* If new_m2 overflowed int32_t before rescaling, preserve proportional value */
            if (new_m2 >= (int64_t)INT32_MAX && new_count >= (4LL * HT_FIXED_ONE)) {
                int64_t c_tmp = new_count;
                int64_t m_tmp = new_m2;
                while ((c_tmp >= (int64_t)HT_FIXED_RESCALE_THRESHOLD ||
                        m_tmp >= (int64_t)HT_FIXED_RESCALE_THRESHOLD) &&
                       c_tmp >= (4LL * HT_FIXED_ONE)) {
                    c_tmp = (int64_t)((uint64_t)c_tmp >> 1);
                    m_tmp = (int64_t)((uint64_t)m_tmp >> 1);
                }
                o->count = ht_fixed_saturate64(c_tmp);
                o->m2 = ht_fixed_saturate64(m_tmp);
            }
        } else {
            o->count = ht_fixed_saturate64(new_count);
            o->m2 = ht_fixed_saturate64(new_m2);
        }
#else
        ht_float_t old_count = o->count;
        ht_float_t new_count = HT_ADD(old_count, weight);
        ht_float_t delta = HT_SUB(val, o->mean);

        o->mean = HT_ADD(o->mean, HT_MUL(HT_DIV(weight, new_count), delta));
        ht_float_t delta2 = HT_SUB(val, o->mean);
        o->m2 = HT_ADD(o->m2, HT_MUL(weight, HT_MUL(delta, delta2)));
        o->count = new_count;
#endif

        if (val < o->min_val) {
            o->min_val = val;
        }
        if (val > o->max_val) {
            o->max_val = val;
        }
    }
}

ht_float_t ht_gaussian_variance(const ht_gaussian_obs_t *obs) {
    if (!obs || obs->count <= HT_CONST(1.0) || obs->m2 <= HT_CONST(0.0)) {
        return HT_CONST(0.0);
    }

    ht_float_t var = HT_DIV(obs->m2, HT_SUB(obs->count, HT_CONST(1.0)));
    return (var > HT_CONST(0.0)) ? var : HT_CONST(0.0);
}

ht_float_t ht_gaussian_stddev(const ht_gaussian_obs_t *obs) {
    ht_float_t var = ht_gaussian_variance(obs);
    if (var > HT_CONST(0.0)) {
        return HT_SQRT(var);
    }
#ifdef HT_USE_FIXED_POINT
    /* Sub-ULP m2 underflow guard: when samples genuinely vary (max_val > min_val)
     * but delta^2 < 0.5 ULP in Q16.16 (e.g. narrow monotonic timestamps like Electricity
     * 'date' with std ~ 0.0006), approximate stddev via uniform range (max-min)/sqrt(12)
     * so ht_normal_cdf does not collapse into a spurious Dirac step function. */
    if (obs && obs->count >= HT_CONST(2.0) &&
        HT_ISFINITE(obs->min_val) && HT_ISFINITE(obs->max_val) &&
        obs->max_val > obs->min_val) {
        ht_float_t range = HT_SUB(obs->max_val, obs->min_val);
        ht_float_t est = HT_MUL(range, HT_CONST(0.288675)); /* 1 / sqrt(12) */
        return (est > HT_CONST(0.0)) ? est : HT_EPSILON;
    }
#endif
    return HT_CONST(0.0);
}

ht_float_t ht_normal_cdf(ht_float_t x, ht_float_t mean, ht_float_t stddev) {
    if (HT_ISNAN(x) || HT_ISNAN(mean)) {
        return HT_NAN_VAL;
    }

    if (stddev <= HT_CONST(0.0) || HT_ISNAN(stddev)) {
        if (x < mean) {
            return HT_CONST(0.0);
        }
        if (x > mean) {
            return HT_CONST(1.0);
        }
        return HT_CONST(0.5);
    }

    ht_float_t z = HT_DIV(HT_SUB(x, mean), HT_MUL(stddev, HT_SQRT2));
    ht_float_t cdf = HT_MUL(HT_CONST(0.5), HT_ADD(HT_CONST(1.0), HT_ERF(z)));

    if (cdf < HT_CONST(0.0)) {
        cdf = HT_CONST(0.0);
    } else if (cdf > HT_CONST(1.0)) {
        cdf = HT_CONST(1.0);
    }

    return cdf;
}

ht_float_t ht_gaussian_cond_proba(const ht_gaussian_obs_t *obs, ht_float_t val) {
    if (!obs || obs->count < HT_CONST(2.0) || HT_ISNAN(val) || !HT_ISFINITE(val)) {
        return HT_CONST(0.0);
    }

#ifdef HT_USE_FIXED_POINT
    ht_float_t diff = HT_SUB(val, obs->mean);
    ht_float_t stddev = ht_gaussian_stddev(obs);
    if (stddev <= HT_CONST(0.0) || HT_ISNAN(stddev)) {
        return HT_CONST(0.0);
    }
    ht_float_t inv_norm = HT_DIV(HT_CONST(1.0), HT_MUL(stddev, HT_SQRT_TWO_PI));
    ht_float_t z = HT_DIV(diff, stddev);
    ht_float_t exponent = HT_MUL(HT_CONST(-0.5), HT_MUL(z, z));

    return HT_MUL(inv_norm, HT_EXP(exponent));
#else
    ht_float_t s2 = ht_gaussian_variance(obs);
    if (s2 <= HT_CONST(0.0) || HT_ISNAN(s2)) {
        return HT_CONST(0.0);
    }

    ht_float_t diff = val - obs->mean;
    ht_float_t denom = HT_SQRT(HT_TWO_PI * s2);
    ht_float_t exponent = -(diff * diff) / ((ht_float_t)2.0 * s2);

    return HT_EXP(exponent) / denom;
#endif
}

ht_float_t ht_gaussian_cond_log_proba(const ht_gaussian_obs_t *obs, ht_float_t val) {
    if (!obs || obs->count < HT_CONST(2.0) || HT_ISNAN(val) || !HT_ISFINITE(val)) {
        return -HT_INFINITY;
    }

#ifdef HT_USE_FIXED_POINT
    ht_float_t stddev = ht_gaussian_stddev(obs);
    if (stddev <= HT_CONST(0.0) || HT_ISNAN(stddev)) {
        return -HT_INFINITY;
    }

    ht_float_t diff = HT_SUB(val, obs->mean);
    /* Factorized log(2*pi*s2) = ln(2*pi) + 2*ln(sigma) and z = diff/sigma to prevent Q16.16 overflow */
    ht_float_t log_two_pi_s2 = HT_ADD(HT_LN_2PI, HT_MUL(HT_CONST(2.0), HT_LOG(stddev)));
    ht_float_t z = HT_DIV(diff, stddev);
    ht_float_t exponent_term = HT_MUL(HT_CONST(0.5), HT_MUL(z, z));

    return HT_SUB(HT_MUL(HT_CONST(-0.5), log_two_pi_s2), exponent_term);
#else
    ht_float_t s2 = ht_gaussian_variance(obs);
    if (s2 <= HT_CONST(0.0) || HT_ISNAN(s2)) {
        return -HT_INFINITY;
    }

    ht_float_t diff = val - obs->mean;
    ht_float_t log_two_pi_s2 = HT_LOG(HT_TWO_PI) + HT_LOG(s2);
    ht_float_t exponent_term = (diff * diff) / ((ht_float_t)2.0 * s2);

    return (ht_float_t)-0.5 * log_two_pi_s2 - exponent_term;
#endif
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
        if (obs[c].count > HT_CONST(0.0)) {
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

    ht_float_t bin_size = HT_DIV(HT_SUB(max_val, min_val), HT_FROM_UINT(HT_GAUSSIAN_SPLITS + 1U));
    uint8_t count = 0;

    for (uint8_t i = 0; i < HT_GAUSSIAN_SPLITS; ++i) {
        ht_float_t split_val = HT_ADD(min_val, HT_MUL(bin_size, HT_FROM_UINT((uint32_t)i + 1U)));
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
            lhs_dist[c] = HT_CONST(0.0);
        }
    }
    if (rhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            rhs_dist[c] = HT_CONST(0.0);
        }
    }

    if (!obs || !lhs_dist || !rhs_dist) {
        return;
    }

    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;

    for (uint8_t c = 0; c < max_c; ++c) {
        if (obs[c].count <= HT_CONST(0.0)) {
            continue;
        }

        if (split_val < obs[c].min_val) {
            rhs_dist[c] = obs[c].count;
        } else if (split_val >= obs[c].max_val) {
            lhs_dist[c] = obs[c].count;
        } else {
            ht_float_t stddev = ht_gaussian_stddev(&obs[c]);
            ht_float_t cdf = ht_normal_cdf(split_val, obs[c].mean, stddev);
            ht_float_t left_w = HT_MUL(cdf, obs[c].count);
            lhs_dist[c] = left_w;
            ht_float_t right_w = HT_SUB(obs[c].count, left_w);
            rhs_dist[c] = (right_w > HT_CONST(0.0)) ? right_w : HT_CONST(0.0);
        }
    }
}

/* ========================================================================= */
/* --- Histogram Numeric Observer Implementation (Ben-Haim & Tom-Tov) ------ */
/* ========================================================================= */

static uint8_t ht_hist_active_bins(const ht_hist_obs_t *obs) {
    uint8_t count = 0;
    for (uint8_t i = 0; i < HT_HIST_MAX_BINS; ++i) {
        ht_float_t total = HT_CONST(0.0);
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            total = HT_ADD(total, obs->bins[i].counts[c]);
        }
        if (total > HT_CONST(0.0)) {
            count++;
        } else {
            break;
        }
    }
    return count;
}

void ht_hist_init(ht_hist_obs_t *obs) {
    if (obs != NULL) {
        memset(obs, 0, sizeof(*obs));
    }
}

void ht_hist_update(ht_hist_obs_t *obs, uint8_t class_idx, ht_float_t val, ht_float_t weight) {
    if (obs == NULL || class_idx >= HT_MAX_CLASSES || HT_ISNAN(val) || weight <= HT_CONST(0.0)) {
        return;
    }

    uint8_t n_bins = ht_hist_active_bins(obs);

    /* Check if an existing bin has the same center */
    for (uint8_t i = 0; i < n_bins; ++i) {
        if (HT_FABS(HT_SUB(obs->bins[i].center, val)) <= HT_EPSILON) {
            obs->bins[i].counts[class_idx] = HT_ADD(obs->bins[i].counts[class_idx], weight);
#ifdef HT_USE_FIXED_POINT
            if (obs->bins[i].counts[class_idx] >= HT_FIXED_RESCALE_THRESHOLD) {
                ht_hist_rescale_all(obs, n_bins);
            }
#endif
            return;
        }
    }

    if (n_bins < HT_HIST_MAX_BINS) {
        uint8_t ins_idx = n_bins;
        for (uint8_t i = 0; i < n_bins; ++i) {
            if (val < obs->bins[i].center) {
                ins_idx = i;
                break;
            }
        }
        for (int32_t j = (int32_t)n_bins - 1; j >= (int32_t)ins_idx; --j) {
            obs->bins[j + 1] = obs->bins[j];
        }
        obs->bins[ins_idx].center = val;
        memset(obs->bins[ins_idx].counts, 0, sizeof(obs->bins[ins_idx].counts));
        obs->bins[ins_idx].counts[class_idx] = weight;
    } else {
        ht_hist_bin_t temp[HT_HIST_MAX_BINS + 1U];
        uint8_t ins_idx = HT_HIST_MAX_BINS;
        for (uint8_t i = 0; i < HT_HIST_MAX_BINS; ++i) {
            if (val < obs->bins[i].center) {
                ins_idx = i;
                break;
            }
        }
        for (uint8_t i = 0; i < ins_idx; ++i) {
            temp[i] = obs->bins[i];
        }
        temp[ins_idx].center = val;
        memset(temp[ins_idx].counts, 0, sizeof(temp[ins_idx].counts));
        temp[ins_idx].counts[class_idx] = weight;
        for (uint8_t i = ins_idx; i < HT_HIST_MAX_BINS; ++i) {
            temp[i + 1U] = obs->bins[i];
        }

        uint8_t min_idx = 0;
        ht_float_t min_dist = HT_SUB(temp[1].center, temp[0].center);
        for (uint8_t i = 1; i < HT_HIST_MAX_BINS; ++i) {
            ht_float_t d = HT_SUB(temp[i + 1U].center, temp[i].center);
            if (d < min_dist) {
                min_dist = d;
                min_idx = i;
            }
        }

        ht_float_t w0 = HT_CONST(0.0);
        ht_float_t w1 = HT_CONST(0.0);
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            w0 = HT_ADD(w0, temp[min_idx].counts[c]);
            w1 = HT_ADD(w1, temp[min_idx + 1U].counts[c]);
        }
        ht_float_t w_comb = HT_ADD(w0, w1);
#ifdef HT_USE_FIXED_POINT
        /* Overflow-safe weighted average: c0 + (w1 / (w0 + w1)) * (c1 - c0) */
        ht_float_t p_comb = temp[min_idx].center;
        if (w_comb > HT_EPSILON) {
            ht_float_t ratio = HT_DIV(w1, w_comb);
            p_comb = HT_ADD(temp[min_idx].center,
                            HT_MUL(ratio, HT_SUB(temp[min_idx + 1U].center, temp[min_idx].center)));
        }
#else
        ht_float_t p_comb = (w_comb > HT_EPSILON)
            ? ((temp[min_idx].center * w0 + temp[min_idx + 1U].center * w1) / w_comb)
            : temp[min_idx].center;
#endif

        for (uint8_t i = 0; i < min_idx; ++i) {
            obs->bins[i] = temp[i];
        }
        obs->bins[min_idx].center = p_comb;
        bool need_rescale = false;
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            obs->bins[min_idx].counts[c] = HT_ADD(temp[min_idx].counts[c], temp[min_idx + 1U].counts[c]);
#ifdef HT_USE_FIXED_POINT
            if (obs->bins[min_idx].counts[c] >= HT_FIXED_RESCALE_THRESHOLD) {
                need_rescale = true;
            }
#endif
        }
        for (uint8_t i = min_idx + 1U; i < HT_HIST_MAX_BINS; ++i) {
            obs->bins[i] = temp[i + 1U];
        }
#ifdef HT_USE_FIXED_POINT
        if (need_rescale) {
            ht_hist_rescale_all(obs, (uint8_t)HT_HIST_MAX_BINS);
        }
#else
        (void)need_rescale;
#endif
    }
}

ht_float_t ht_hist_cond_proba(const ht_hist_obs_t *obs, uint8_t class_idx, ht_float_t val) {
    if (obs == NULL || class_idx >= HT_MAX_CLASSES || HT_ISNAN(val)) {
        return HT_CONST(0.0);
    }

    uint8_t n_bins = ht_hist_active_bins(obs);
    if (n_bins == 0) return HT_CONST(0.0);

    ht_float_t total_c = HT_CONST(0.0);
    for (uint8_t i = 0; i < n_bins; ++i) {
        total_c = HT_ADD(total_c, obs->bins[i].counts[class_idx]);
    }
    if (total_c <= HT_CONST(0.0)) return HT_CONST(0.0);

    if (n_bins == 1U) {
        ht_float_t diff = HT_FABS(HT_SUB(val, obs->bins[0].center));
        return (diff <= HT_CONST(1.0)) ? HT_DIV(obs->bins[0].counts[class_idx], total_c) : HT_CONST(0.0);
    }

    ht_float_t min_p = obs->bins[0].center;
    ht_float_t max_p = obs->bins[n_bins - 1U].center;
    ht_float_t span = HT_SUB(max_p, min_p);
#ifdef HT_USE_FIXED_POINT
    ht_float_t span_eps = HT_EPSILON;
    ht_float_t min_d = HT_CONST(1e-4);
#else
    ht_float_t span_eps = (ht_float_t)1e-9;
    ht_float_t min_d = (ht_float_t)1e-6;
#endif
    if (span <= span_eps) {
        return HT_DIV(obs->bins[0].counts[class_idx], total_c);
    }

    if (val <= min_p) {
        ht_float_t d = HT_SUB(obs->bins[1].center, obs->bins[0].center);
        return HT_DIV(HT_DIV(obs->bins[0].counts[class_idx], total_c), (d > min_d) ? d : HT_CONST(1.0));
    }
    if (val >= max_p) {
        ht_float_t d = HT_SUB(obs->bins[n_bins - 1U].center, obs->bins[n_bins - 2U].center);
        return HT_DIV(HT_DIV(obs->bins[n_bins - 1U].counts[class_idx], total_c), (d > min_d) ? d : HT_CONST(1.0));
    }

    uint8_t seg = 0;
    for (uint8_t i = 0; i < n_bins - 1U; ++i) {
        if (val >= obs->bins[i].center && val <= obs->bins[i + 1U].center) {
            seg = i;
            break;
        }
    }

    ht_float_t d_seg = HT_SUB(obs->bins[seg + 1U].center, obs->bins[seg].center);
    if (d_seg <= span_eps) d_seg = min_d;
    ht_float_t alpha = HT_DIV(HT_SUB(val, obs->bins[seg].center), d_seg);
    ht_float_t dens0 = HT_DIV(HT_DIV(obs->bins[seg].counts[class_idx], total_c), d_seg);
    ht_float_t dens1 = HT_DIV(HT_DIV(obs->bins[seg + 1U].counts[class_idx], total_c), d_seg);
    return HT_ADD(HT_MUL(HT_SUB(HT_CONST(1.0), alpha), dens0), HT_MUL(alpha, dens1));
}

ht_float_t ht_hist_cond_log_proba(const ht_hist_obs_t *obs, uint8_t class_idx, ht_float_t val) {
    ht_float_t p = ht_hist_cond_proba(obs, class_idx, val);
    if (p <= HT_CONST(0.0)) {
        return -HT_INFINITY;
    }
    return HT_LOG(p);
}

uint8_t ht_hist_split_suggestions(const ht_hist_obs_t *obs, uint8_t n_classes, ht_float_t splits_out[HT_GAUSSIAN_SPLITS]) {
    (void)n_classes;
    if (obs == NULL || splits_out == NULL) return 0;
    uint8_t n_bins = ht_hist_active_bins(obs);
    if (n_bins < 2U) return 0;

    uint8_t count = 0;
    for (uint8_t i = 0; i < n_bins - 1U && count < HT_GAUSSIAN_SPLITS; ++i) {
        ht_float_t mid = HT_MUL(HT_ADD(obs->bins[i].center, obs->bins[i + 1U].center), HT_CONST(0.5));
        splits_out[count++] = mid;
    }
    return count;
}

void ht_hist_binary_split_dists(const ht_hist_obs_t *obs,
                                uint8_t n_classes,
                                ht_float_t split_val,
                                ht_float_t lhs_dist[HT_MAX_CLASSES],
                                ht_float_t rhs_dist[HT_MAX_CLASSES])
{
    if (lhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) lhs_dist[c] = HT_CONST(0.0);
    }
    if (rhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) rhs_dist[c] = HT_CONST(0.0);
    }
    if (!obs || !lhs_dist || !rhs_dist) return;

    uint8_t n_bins = ht_hist_active_bins(obs);
    if (n_bins == 0) return;

    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;

    for (uint8_t c = 0; c < max_c; ++c) {
        ht_float_t left_w = HT_CONST(0.0);
        ht_float_t right_w = HT_CONST(0.0);
        for (uint8_t i = 0; i < n_bins; ++i) {
            if (obs->bins[i].center <= split_val) {
                left_w = HT_ADD(left_w, obs->bins[i].counts[c]);
            } else {
                right_w = HT_ADD(right_w, obs->bins[i].counts[c]);
            }
        }
        lhs_dist[c] = left_w;
        rhs_dist[c] = right_w;
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
        cat_val >= (int32_t)HT_MAX_CATEGORIES || weight <= HT_CONST(0.0)) {
        return;
    }

    obs->counts[class_idx][cat_val] = HT_ADD(obs->counts[class_idx][cat_val], weight);
#ifdef HT_USE_FIXED_POINT
    if (obs->counts[class_idx][cat_val] >= HT_FIXED_RESCALE_THRESHOLD) {
        ht_categorical_rescale_all(obs);
    }
#endif
}

ht_float_t ht_categorical_cond_proba(const ht_categorical_obs_t *obs,
                                     uint8_t class_idx,
                                     int32_t cat_val)
{
    if (!obs || class_idx >= HT_MAX_CLASSES || cat_val < 0 ||
        cat_val >= (int32_t)HT_MAX_CATEGORIES) {
        return HT_CONST(0.0);
    }

#ifdef HT_USE_FIXED_POINT
    int64_t total64 = 0;
    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
        total64 += (int64_t)obs->counts[class_idx][v];
    }
    if (total64 <= 0) {
        return 0;
    }
    uint64_t num = ((uint64_t)(uint32_t)obs->counts[class_idx][cat_val] << 16) + ((uint64_t)total64 >> 1);
    return ht_fixed_saturate64((int64_t)(num / (uint64_t)total64));
#else
    ht_float_t total = HT_CONST(0.0);
    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
        total = HT_ADD(total, obs->counts[class_idx][v]);
    }

    if (total <= HT_CONST(0.0)) {
        return HT_CONST(0.0);
    }

    return HT_DIV(obs->counts[class_idx][cat_val], total);
#endif
}

bool ht_categorical_is_observed(const ht_categorical_obs_t *obs, int32_t cat_val) {
    if (!obs || cat_val < 0 || cat_val >= (int32_t)HT_MAX_CATEGORIES) {
        return false;
    }

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        if (obs->counts[c][cat_val] > HT_CONST(0.0)) {
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
            lhs_dist[c] = HT_CONST(0.0);
        }
    }
    if (rhs_dist) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            rhs_dist[c] = HT_CONST(0.0);
        }
    }

    if (!obs || !lhs_dist || !rhs_dist) {
        return;
    }

    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;

    for (uint8_t c = 0; c < max_c; ++c) {
        ht_float_t eq_w = HT_CONST(0.0);
        ht_float_t not_eq_w = HT_CONST(0.0);

        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            ht_float_t w = obs->counts[c][v];
            if (cat_val >= 0 && v == (uint8_t)cat_val) {
                eq_w = HT_ADD(eq_w, w);
            } else {
                not_eq_w = HT_ADD(not_eq_w, w);
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
        cand->lhs_dist[c] = HT_CONST(0.0);
        cand->rhs_dist[c] = HT_CONST(0.0);
    }
    cand->split_val.threshold = HT_CONST(0.0);
    cand->feature_idx = HT_NO_FEATURE;
    cand->is_categorical = 0;
}

/* ========================================================================= */
/* --- Incremental Variance and Regression Observer Implementation --------- */
/* ========================================================================= */

void ht_var_init(ht_var_stat_t *var) {
    if (var) {
        var->count = HT_CONST(0.0);
        var->mean = HT_CONST(0.0);
        var->m2 = HT_CONST(0.0);
    }
}

void ht_var_update(ht_var_stat_t *var, ht_float_t y, ht_float_t weight) {
    if (!var || weight <= HT_CONST(0.0) || HT_ISNAN(y) || !HT_ISFINITE(y)) {
        return;
    }

    if (var->count <= HT_CONST(0.0)) {
        var->count = weight;
        var->mean = y;
        var->m2 = HT_CONST(0.0);
    } else {
#ifdef HT_USE_FIXED_POINT
        /* var->count and var->mean are Q16.16; var->m2 is stored in Q24.8 (scale 256, max ~8.38M) */
        int64_t old_count = (int64_t)var->count;
        int64_t w64 = (int64_t)weight;
        int64_t new_count = old_count + w64;
        int64_t delta = (int64_t)y - (int64_t)var->mean;
        int64_t new_mean = (int64_t)var->mean + ((delta * w64) / new_count);
        int64_t delta2 = (int64_t)y - new_mean;

        int64_t d1d2_q8 = ht_q16_sq_to_q8(delta, delta2);
        int64_t inc_m2_q8 = ht_q8_mul_q16_to_q8(d1d2_q8, w64);
        int64_t new_m2 = (int64_t)var->m2 + inc_m2_q8;

        /* Block-Rescaling in 64-bit before storing back (preserves count >= 2.0) */
        while ((new_count >= (int64_t)HT_FIXED_RESCALE_THRESHOLD ||
                new_m2 >= (int64_t)HT_FIXED_M2_RESCALE_THR) &&
               new_count >= (4LL * HT_FIXED_ONE)) {
            new_count = (int64_t)((uint64_t)new_count >> 1);
            new_m2 = (new_m2 > 0) ? (int64_t)((uint64_t)new_m2 >> 1) : 0;
        }

        var->count = ht_fixed_saturate64(new_count);
        var->mean = ht_fixed_saturate64(new_mean);
        var->m2 = (new_m2 > 0) ? ht_fixed_saturate64(new_m2) : 0;
#else
        ht_float_t old_count = var->count;
        ht_float_t new_count = HT_ADD(old_count, weight);
        ht_float_t delta = HT_SUB(y, var->mean);

        var->mean = HT_ADD(var->mean, HT_MUL(HT_DIV(weight, new_count), delta));
        ht_float_t delta2 = HT_SUB(y, var->mean);
        var->m2 = HT_ADD(var->m2, HT_MUL(weight, HT_MUL(delta, delta2)));
        var->count = new_count;

        if (var->m2 < HT_CONST(0.0)) {
            var->m2 = HT_CONST(0.0);
        }
#endif
    }
}

#ifdef HT_USE_FIXED_POINT
int64_t ht_var_value_q8(const ht_var_stat_t *var) {
    if (!var || var->count <= HT_FIXED_ONE || var->m2 <= 0) {
        return 0;
    }
    uint64_t denom = (uint64_t)((int64_t)var->count - (int64_t)HT_FIXED_ONE);
    uint64_t num = (((uint64_t)(uint32_t)var->m2) << 16) + (denom >> 1);
    return (int64_t)(num / denom);
}
#endif

ht_float_t ht_var_value(const ht_var_stat_t *var) {
    if (!var || var->count <= HT_CONST(1.0) || var->m2 <= HT_CONST(0.0)) {
        return HT_CONST(0.0);
    }
#ifdef HT_USE_FIXED_POINT
    int64_t v_q8 = ht_var_value_q8(var);
    if (v_q8 <= 0) {
        return 0;
    }
    const unsigned int lshift = (unsigned int)(HT_FIXED_SHIFT - HT_FIXED_M2_SHIFT);
    if (v_q8 >= ((int64_t)INT32_MAX >> lshift)) {
        return INT32_MAX;
    }
    return (ht_float_t)((uint64_t)v_q8 << lshift);
#else
    ht_float_t v = HT_DIV(var->m2, HT_SUB(var->count, HT_CONST(1.0)));
    return (v > HT_CONST(0.0)) ? v : HT_CONST(0.0);
#endif
}

ht_var_stat_t ht_var_combine(const ht_var_stat_t *a, const ht_var_stat_t *b) {
    ht_var_stat_t res;
    res.count = HT_CONST(0.0);
    res.mean = HT_CONST(0.0);
    res.m2 = HT_CONST(0.0);

    bool a_valid = (a && a->count > HT_CONST(0.0));
    bool b_valid = (b && b->count > HT_CONST(0.0));

    if (!a_valid && !b_valid) {
        return res;
    }
    if (!a_valid) {
        return *b;
    }
    if (!b_valid) {
        return *a;
    }

#ifdef HT_USE_FIXED_POINT
    int64_t na = (int64_t)a->count;
    int64_t nb = (int64_t)b->count;
    int64_t nc = na + nb;
    int64_t delta = (int64_t)b->mean - (int64_t)a->mean;
    int64_t mean_c = (int64_t)a->mean + ((delta * nb) / nc);
    int64_t delta_sq_q8 = ht_q16_sq_to_q8(delta, delta);
    int64_t weight_term = (na * nb) / nc; /* Q16.16 */
    int64_t m2_c = (int64_t)a->m2 + (int64_t)b->m2 + ht_q8_mul_q16_to_q8(delta_sq_q8, weight_term);

    while ((nc >= (int64_t)HT_FIXED_RESCALE_THRESHOLD ||
            m2_c >= (int64_t)HT_FIXED_M2_RESCALE_THR) &&
           nc >= (4LL * HT_FIXED_ONE)) {
        nc = (int64_t)((uint64_t)nc >> 1);
        m2_c = (m2_c > 0) ? (int64_t)((uint64_t)m2_c >> 1) : 0;
    }

    res.count = ht_fixed_saturate64(nc);
    res.mean = ht_fixed_saturate64(mean_c);
    res.m2 = (m2_c > 0) ? ht_fixed_saturate64(m2_c) : 0;
#else
    res.count = HT_ADD(a->count, b->count);
    ht_float_t delta = HT_SUB(b->mean, a->mean);
    res.mean = HT_ADD(a->mean, HT_MUL(HT_DIV(b->count, res.count), delta));
    res.m2 = HT_ADD(HT_ADD(a->m2, b->m2),
                    HT_MUL(HT_MUL(delta, delta), HT_MUL(a->count, HT_DIV(b->count, res.count))));

    if (res.m2 < HT_CONST(0.0)) {
        res.m2 = HT_CONST(0.0);
    }
#endif

    return res;
}

ht_var_stat_t ht_var_subtract(const ht_var_stat_t *total, const ht_var_stat_t *part) {
    ht_var_stat_t res;
    res.count = HT_CONST(0.0);
    res.mean = HT_CONST(0.0);
    res.m2 = HT_CONST(0.0);

    if (!total || total->count <= HT_CONST(0.0)) {
        return res;
    }
    if (!part || part->count <= HT_CONST(0.0)) {
        return *total;
    }

    if (part->count >= total->count) {
        return res;
    }

#ifdef HT_USE_FIXED_POINT
    int64_t nt = (int64_t)total->count;
    int64_t np = (int64_t)part->count;
    int64_t nr = nt - np;
    if (nr <= 0) {
        return res;
    }
    /* Exact 64-bit mean subtraction: mean_r = total->mean + (np / nr) * (total->mean - part->mean) */
    int64_t mean_r = (int64_t)total->mean + (((int64_t)total->mean - (int64_t)part->mean) * np) / nr;
    int64_t delta = mean_r - (int64_t)part->mean;
    int64_t delta_sq_q8 = ht_q16_sq_to_q8(delta, delta);
    int64_t weight_term = (nr * np) / nt;
    int64_t diff_m2 = (int64_t)total->m2 - (int64_t)part->m2 - ht_q8_mul_q16_to_q8(delta_sq_q8, weight_term);

    res.count = ht_fixed_saturate64(nr);
    res.mean = ht_fixed_saturate64(mean_r);
    res.m2 = (diff_m2 > 0) ? ht_fixed_saturate64(diff_m2) : 0;
#else
    res.count = HT_SUB(total->count, part->count);
    if (res.count <= HT_CONST(0.0)) {
        return res;
    }

    res.mean = HT_DIV(HT_SUB(HT_MUL(total->count, total->mean), HT_MUL(part->count, part->mean)), res.count);
    ht_float_t delta = HT_SUB(res.mean, part->mean);
    ht_float_t diff_m2 = HT_SUB(HT_SUB(total->m2, part->m2),
                                HT_MUL(HT_MUL(delta, delta), HT_MUL(res.count, HT_DIV(part->count, total->count))));

    res.m2 = (diff_m2 > HT_CONST(0.0)) ? diff_m2 : HT_CONST(0.0);
#endif

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
    if (!obs || weight <= HT_CONST(0.0) ||
        HT_ISNAN(x_val) || !HT_ISFINITE(x_val) ||
        HT_ISNAN(y_val) || !HT_ISFINITE(y_val)) {
        return;
    }

    /* Count active slots in obs->slots (slots are strictly packed 0..m-1) */
    uint8_t m = 0;
    while (m < HT_REG_NUM_SLOTS && obs->slots[m].y_stats.count > HT_CONST(0.0)) {
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
        obs->slots[pos].y_stats.m2 = HT_CONST(0.0);
        return;
    }

    /* All HT_REG_NUM_SLOTS slots are occupied.
     * Find adjacent pair with minimum distance between their x_means. */
    uint8_t min_idx = 0;
    ht_float_t min_dist = HT_SUB(obs->slots[1].x_mean, obs->slots[0].x_mean);
    for (uint8_t i = 1; i < HT_REG_NUM_SLOTS - 1; ++i) {
        ht_float_t dist = HT_SUB(obs->slots[i + 1].x_mean, obs->slots[i].x_mean);
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }

    /* Fuse slots min_idx and min_idx + 1 using Chan's parallel formulas */
    ht_reg_num_slot_t *s0 = &obs->slots[min_idx];
    ht_reg_num_slot_t *s1 = &obs->slots[min_idx + 1];

#ifdef HT_USE_FIXED_POINT
    int64_t w0 = (int64_t)s0->y_stats.count;
    int64_t w1 = (int64_t)s1->y_stats.count;
    int64_t wsum = w0 + w1;
    ht_float_t fused_x = (wsum > 0)
        ? ht_fixed_saturate64((int64_t)s0->x_mean + (((int64_t)s1->x_mean - (int64_t)s0->x_mean) * w1) / wsum)
        : s0->x_mean;
#else
    ht_float_t fused_count = s0->y_stats.count + s1->y_stats.count;
    ht_float_t fused_x = (s0->y_stats.count * s0->x_mean + s1->y_stats.count * s1->x_mean) / fused_count;
#endif
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
    obs->slots[pos].y_stats.m2 = HT_CONST(0.0);
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
        weight <= HT_CONST(0.0) || HT_ISNAN(y_val) || !HT_ISFINITE(y_val)) {
        return;
    }

    ht_var_update(&obs->cat_stats[cat_val], y_val, weight);
}

/* ========================================================================= */
/* --- Sufficient Statistics Merge Implementation -------------------------- */
/* ========================================================================= */

void ht_gaussian_merge(ht_gaussian_obs_t *dst, const ht_gaussian_obs_t *src) {
    if (!dst || !src || src->count <= HT_CONST(0.0)) {
        return;
    }
    if (dst->count <= HT_CONST(0.0)) {
        *dst = *src;
        return;
    }

#ifdef HT_USE_FIXED_POINT
    int64_t n_a = (int64_t)dst->count;
    int64_t n_b = (int64_t)src->count;
    int64_t n_c = n_a + n_b;
    int64_t delta = (int64_t)src->mean - (int64_t)dst->mean;
    int64_t mean_c = (int64_t)dst->mean + ((delta * n_b) / n_c);
    int64_t d2_q16 = (int64_t)(((uint64_t)(delta * delta) + 32768ULL) >> 16);
    int64_t w_term = (n_a * n_b) / n_c;
    int64_t cross = (d2_q16 <= (1LL << 32))
        ? (int64_t)(((uint64_t)d2_q16 * (uint64_t)w_term + 32768ULL) >> 16)
        : (int64_t)(((uint64_t)d2_q16 >> 8) * ((uint64_t)w_term >> 8));
    int64_t m2_c = (int64_t)dst->m2 + (int64_t)src->m2 + cross;

    while ((n_c >= (int64_t)HT_FIXED_RESCALE_THRESHOLD ||
            m2_c >= (int64_t)HT_FIXED_RESCALE_THRESHOLD) &&
           n_c >= (4LL * HT_FIXED_ONE)) {
        n_c = (int64_t)((uint64_t)n_c >> 1);
        m2_c = (m2_c > 0) ? (int64_t)((uint64_t)m2_c >> 1) : 0;
    }

    dst->mean = ht_fixed_saturate64(mean_c);
    dst->m2 = (m2_c > 0) ? ht_fixed_saturate64(m2_c) : 0;
    dst->count = ht_fixed_saturate64(n_c);
#else
    ht_float_t n_a = dst->count;
    ht_float_t n_b = src->count;
    ht_float_t n_c = HT_ADD(n_a, n_b);
    ht_float_t delta = HT_SUB(src->mean, dst->mean);

    dst->mean = HT_ADD(dst->mean, HT_MUL(HT_DIV(n_b, n_c), delta));
    dst->m2 = HT_ADD(HT_ADD(dst->m2, src->m2),
                     HT_MUL(HT_MUL(delta, delta), HT_MUL(n_a, HT_DIV(n_b, n_c))));
    if (dst->m2 < HT_CONST(0.0)) {
        dst->m2 = HT_CONST(0.0);
    }
    dst->count = n_c;
#endif
    if (src->min_val < dst->min_val) {
        dst->min_val = src->min_val;
    }
    if (src->max_val > dst->max_val) {
        dst->max_val = src->max_val;
    }
}

void ht_categorical_merge(ht_categorical_obs_t *dst,
                          const ht_categorical_obs_t *src,
                          uint8_t n_classes) {
    if (!dst || !src || n_classes == 0U) {
        return;
    }
    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;
    bool need_rescale = false;
    for (uint8_t c = 0; c < max_c; ++c) {
        for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
            dst->counts[c][v] = HT_ADD(dst->counts[c][v], src->counts[c][v]);
#ifdef HT_USE_FIXED_POINT
            if (dst->counts[c][v] >= HT_FIXED_RESCALE_THRESHOLD) {
                need_rescale = true;
            }
#endif
        }
    }
#ifdef HT_USE_FIXED_POINT
    if (need_rescale) {
        ht_categorical_rescale_all(dst);
    }
#else
    (void)need_rescale;
#endif
}

static void ht_hist_insert_bin(ht_hist_obs_t *dst,
                               const ht_hist_bin_t *in_bin,
                               uint8_t max_c) {
    ht_float_t in_total = HT_CONST(0.0);
    for (uint8_t c = 0; c < max_c; ++c) {
        in_total = HT_ADD(in_total, in_bin->counts[c]);
    }
    if (in_total <= HT_CONST(0.0) || HT_ISNAN(in_bin->center)) {
        return;
    }

    uint8_t n_bins = ht_hist_active_bins(dst);

    for (uint8_t i = 0; i < n_bins; ++i) {
        if (HT_FABS(HT_SUB(dst->bins[i].center, in_bin->center)) <= HT_EPSILON) {
            bool need_rescale = false;
            for (uint8_t c = 0; c < max_c; ++c) {
                dst->bins[i].counts[c] = HT_ADD(dst->bins[i].counts[c], in_bin->counts[c]);
#ifdef HT_USE_FIXED_POINT
                if (dst->bins[i].counts[c] >= HT_FIXED_RESCALE_THRESHOLD) {
                    need_rescale = true;
                }
#endif
            }
#ifdef HT_USE_FIXED_POINT
            if (need_rescale) {
                ht_hist_rescale_all(dst, n_bins);
            }
#else
            (void)need_rescale;
#endif
            return;
        }
    }

    if (n_bins < HT_HIST_MAX_BINS) {
        uint8_t ins_idx = n_bins;
        for (uint8_t i = 0; i < n_bins; ++i) {
            if (in_bin->center < dst->bins[i].center) {
                ins_idx = i;
                break;
            }
        }
        for (int32_t j = (int32_t)n_bins - 1; j >= (int32_t)ins_idx; --j) {
            dst->bins[j + 1] = dst->bins[j];
        }
        dst->bins[ins_idx].center = in_bin->center;
        memset(dst->bins[ins_idx].counts, 0, sizeof(dst->bins[ins_idx].counts));
        for (uint8_t c = 0; c < max_c; ++c) {
            dst->bins[ins_idx].counts[c] = in_bin->counts[c];
        }
    } else {
        ht_hist_bin_t temp[HT_HIST_MAX_BINS + 1U];
        uint8_t ins_idx = HT_HIST_MAX_BINS;
        for (uint8_t i = 0; i < HT_HIST_MAX_BINS; ++i) {
            if (in_bin->center < dst->bins[i].center) {
                ins_idx = i;
                break;
            }
        }
        for (uint8_t i = 0; i < ins_idx; ++i) {
            temp[i] = dst->bins[i];
        }
        temp[ins_idx].center = in_bin->center;
        memset(temp[ins_idx].counts, 0, sizeof(temp[ins_idx].counts));
        for (uint8_t c = 0; c < max_c; ++c) {
            temp[ins_idx].counts[c] = in_bin->counts[c];
        }
        for (uint8_t i = ins_idx; i < HT_HIST_MAX_BINS; ++i) {
            temp[i + 1U] = dst->bins[i];
        }

        uint8_t min_idx = 0;
        ht_float_t min_dist = HT_SUB(temp[1].center, temp[0].center);
        for (uint8_t i = 1; i < HT_HIST_MAX_BINS; ++i) {
            ht_float_t d = HT_SUB(temp[i + 1U].center, temp[i].center);
            if (d < min_dist) {
                min_dist = d;
                min_idx = i;
            }
        }

        ht_float_t w0 = HT_CONST(0.0);
        ht_float_t w1 = HT_CONST(0.0);
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            w0 = HT_ADD(w0, temp[min_idx].counts[c]);
            w1 = HT_ADD(w1, temp[min_idx + 1U].counts[c]);
        }
        ht_float_t w_comb = HT_ADD(w0, w1);
#ifdef HT_USE_FIXED_POINT
        ht_float_t p_comb = temp[min_idx].center;
        if (w_comb > HT_EPSILON) {
            ht_float_t ratio = HT_DIV(w1, w_comb);
            p_comb = HT_ADD(temp[min_idx].center,
                            HT_MUL(ratio, HT_SUB(temp[min_idx + 1U].center, temp[min_idx].center)));
        }
#else
        ht_float_t p_comb = (w_comb > HT_EPSILON)
            ? ((temp[min_idx].center * w0 + temp[min_idx + 1U].center * w1) / w_comb)
            : temp[min_idx].center;
#endif

        for (uint8_t i = 0; i < min_idx; ++i) {
            dst->bins[i] = temp[i];
        }
        dst->bins[min_idx].center = p_comb;
        bool need_rescale = false;
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            dst->bins[min_idx].counts[c] = HT_ADD(temp[min_idx].counts[c], temp[min_idx + 1U].counts[c]);
#ifdef HT_USE_FIXED_POINT
            if (dst->bins[min_idx].counts[c] >= HT_FIXED_RESCALE_THRESHOLD) {
                need_rescale = true;
            }
#endif
        }
        for (uint8_t i = min_idx + 1U; i < HT_HIST_MAX_BINS; ++i) {
            dst->bins[i] = temp[i + 1U];
        }
#ifdef HT_USE_FIXED_POINT
        if (need_rescale) {
            ht_hist_rescale_all(dst, (uint8_t)HT_HIST_MAX_BINS);
        }
#else
        (void)need_rescale;
#endif
    }
}

void ht_hist_merge(ht_hist_obs_t *dst,
                   const ht_hist_obs_t *src,
                   uint8_t n_classes) {
    if (!dst || !src || n_classes == 0U) {
        return;
    }
    uint8_t max_c = (n_classes < HT_MAX_CLASSES) ? n_classes : (uint8_t)HT_MAX_CLASSES;
    uint8_t src_bins = ht_hist_active_bins(src);
    for (uint8_t b = 0; b < src_bins; ++b) {
        ht_hist_insert_bin(dst, &src->bins[b], max_c);
    }
}

static void ht_reg_numeric_insert_slot(ht_reg_numeric_obs_t *dst,
                                       const ht_reg_num_slot_t *in_slot) {
    if (in_slot->y_stats.count <= HT_CONST(0.0) || HT_ISNAN(in_slot->x_mean)) {
        return;
    }

    uint8_t m = 0;
    while (m < HT_REG_NUM_SLOTS && dst->slots[m].y_stats.count > HT_CONST(0.0)) {
        m++;
    }

    if (m < HT_REG_NUM_SLOTS) {
        uint8_t pos = 0;
        while (pos < m && dst->slots[pos].x_mean <= in_slot->x_mean) {
            pos++;
        }
        for (uint8_t i = m; i > pos; --i) {
            dst->slots[i] = dst->slots[i - 1U];
        }
        dst->slots[pos] = *in_slot;
        return;
    }

    ht_reg_num_slot_t temp[HT_REG_NUM_SLOTS + 1U];
    uint8_t pos = 0;
    while (pos < HT_REG_NUM_SLOTS && dst->slots[pos].x_mean <= in_slot->x_mean) {
        pos++;
    }
    for (uint8_t i = 0; i < pos; ++i) {
        temp[i] = dst->slots[i];
    }
    temp[pos] = *in_slot;
    for (uint8_t i = pos; i < HT_REG_NUM_SLOTS; ++i) {
        temp[i + 1U] = dst->slots[i];
    }

    uint8_t min_idx = 0;
    ht_float_t min_dist = HT_SUB(temp[1].x_mean, temp[0].x_mean);
    for (uint8_t i = 1; i < HT_REG_NUM_SLOTS; ++i) {
        ht_float_t d = HT_SUB(temp[i + 1U].x_mean, temp[i].x_mean);
        if (d < min_dist) {
            min_dist = d;
            min_idx = i;
        }
    }

#ifdef HT_USE_FIXED_POINT
    int64_t w0 = (int64_t)temp[min_idx].y_stats.count;
    int64_t w1 = (int64_t)temp[min_idx + 1U].y_stats.count;
    int64_t w_sum = w0 + w1;
    ht_float_t fused_x = (w_sum > 0)
        ? ht_fixed_saturate64((int64_t)temp[min_idx].x_mean +
                              (((int64_t)temp[min_idx + 1U].x_mean - (int64_t)temp[min_idx].x_mean) * w1) / w_sum)
        : temp[min_idx].x_mean;
#else
    ht_float_t w0 = temp[min_idx].y_stats.count;
    ht_float_t w1 = temp[min_idx + 1U].y_stats.count;
    ht_float_t w_sum = w0 + w1;
    ht_float_t fused_x = (w_sum > (ht_float_t)0.0)
        ? ((w0 * temp[min_idx].x_mean + w1 * temp[min_idx + 1U].x_mean) / w_sum)
        : temp[min_idx].x_mean;
#endif
    ht_var_stat_t fused_y = ht_var_combine(&temp[min_idx].y_stats, &temp[min_idx + 1U].y_stats);

    for (uint8_t i = 0; i < min_idx; ++i) {
        dst->slots[i] = temp[i];
    }
    dst->slots[min_idx].x_mean = fused_x;
    dst->slots[min_idx].y_stats = fused_y;
    for (uint8_t i = min_idx + 1U; i < HT_REG_NUM_SLOTS; ++i) {
        dst->slots[i] = temp[i + 1U];
    }
}

void ht_reg_numeric_merge(ht_reg_numeric_obs_t *dst,
                          const ht_reg_numeric_obs_t *src) {
    if (!dst || !src) {
        return;
    }
    for (uint8_t s = 0; s < HT_REG_NUM_SLOTS; ++s) {
        if (src->slots[s].y_stats.count > HT_CONST(0.0)) {
            ht_reg_numeric_insert_slot(dst, &src->slots[s]);
        }
    }
}

void ht_reg_categorical_merge(ht_reg_categorical_obs_t *dst,
                              const ht_reg_categorical_obs_t *src) {
    if (!dst || !src) {
        return;
    }
    for (uint8_t v = 0; v < HT_MAX_CATEGORIES; ++v) {
        dst->cat_stats[v] = ht_var_combine(&dst->cat_stats[v], &src->cat_stats[v]);
    }
}
