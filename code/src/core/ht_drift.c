/**
 * @file ht_drift.c
 * @brief Implementation of the ADWIN Concept Drift Detector and Pool Allocator.
 *
 * Supports float32, double64, and Q16.16 zero-libm fixed-point arithmetic
 * (with 64-bit uint32_t window sizing and Q30 harmonic mean to avoid overflow/underflow).
 */

#include "ht_drift.h"

#ifdef HT_USE_FIXED_POINT
/* Divides a Q16.16 value by a positive uint32_t count without converting count to Q16.16 */
static inline ht_float_t ht_adwin_div_u32(ht_float_t num_q16, uint32_t den) {
    if (den == 0U || HT_ISNAN(num_q16)) {
        return HT_NAN_VAL;
    }
    if (num_q16 == 0) {
        return 0;
    }
    bool neg = (num_q16 < 0);
    uint64_t un = neg ? (uint64_t)(-(int64_t)num_q16) : (uint64_t)num_q16;
    uint64_t ud = (uint64_t)den;
    uint64_t q = (un + (ud >> 1)) / ud;
    if (q >= (uint64_t)INT32_MAX) {
        return neg ? -INT32_MAX : INT32_MAX;
    }
    return neg ? -(ht_float_t)q : (ht_float_t)q;
}

/* Computes ((n1 * n2) / (n1 + n2)) * diff^2 in Q16.16 without overflow for large uint32_t n1, n2 */
static inline ht_float_t ht_adwin_inc_var_u32(uint32_t n1, uint32_t n2, ht_float_t diff_q16) {
    if (n1 == 0U || n2 == 0U || diff_q16 == 0) {
        return 0;
    }
    uint64_t ad = (diff_q16 < 0) ? (uint64_t)(-(int64_t)diff_q16) : (uint64_t)diff_q16;
    uint64_t d2_q16 = (ad * ad + 32768ULL) >> 16; /* Q16.16 */
    uint64_t prod_n = (uint64_t)n1 * (uint64_t)n2;
    uint64_t sum_n = (uint64_t)n1 + (uint64_t)n2;
    /* Compute (d2_q16 * prod_n) / sum_n safely without 64-bit overflow */
    uint64_t w_int = prod_n / sum_n;
    uint64_t w_rem = prod_n % sum_n;
    uint64_t res = 0ULL;
    if (d2_q16 > 0ULL && w_int > ((uint64_t)INT32_MAX / d2_q16)) {
        return INT32_MAX;
    }
    res = d2_q16 * w_int;
    if (d2_q16 <= (UINT64_MAX / (sum_n + 1ULL))) {
        res += (d2_q16 * w_rem + (sum_n >> 1)) / sum_n;
    } else {
        res += (d2_q16 / sum_n) * w_rem;
    }
    return (res >= (uint64_t)INT32_MAX) ? INT32_MAX : (ht_float_t)res;
}

/* Computes ln(n) in Q16.16 for any positive uint32_t n (even n >= 32768) */
static inline ht_float_t ht_adwin_ln_u32(uint32_t n) {
    if (n <= 1U) {
        return 0;
    }
    int32_t shift_bits = 0;
    uint32_t scaled = n;
    while (scaled >= 32768U) {
        scaled >>= 1;
        shift_bits++;
    }
    ht_float_t ln_scaled = HT_LOG(HT_FROM_UINT(scaled));
    if (shift_bits > 0) {
        return HT_ADD(ln_scaled, HT_MUL(HT_FROM_INT(shift_bits), HT_LN2));
    }
    return ln_scaled;
}
#endif

/* Populates default ADWIN configuration matching River defaults */
void ht_adwin_default_config(ht_adwin_config_t *cfg) {
    if (cfg == NULL) {
        return;
    }
    cfg->delta = HT_CONST(0.002);
    cfg->clock = 32U;
    cfg->min_window_length = 5U;
    cfg->grace_period = 10U;
    cfg->max_buckets = HT_ADWIN_MAX_BUCKETS;
    cfg->pad = 0U;
}

/* Initializes an ADWIN instance */
ht_status_t ht_adwin_init(ht_adwin_t *adwin, const ht_adwin_config_t *cfg) {
    if (adwin == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    ht_adwin_config_t default_cfg;
    if (cfg == NULL) {
        ht_adwin_default_config(&default_cfg);
        cfg = &default_cfg;
    } else {
        if (cfg->delta <= HT_CONST(0.0) || cfg->delta >= HT_CONST(1.0) ||
            cfg->clock == 0U || cfg->min_window_length == 0U ||
            cfg->max_buckets == 0U || cfg->max_buckets > HT_ADWIN_MAX_BUCKETS) {
            return HT_ERR_INVALID_ARG;
        }
    }

    memset(adwin, 0, sizeof(*adwin));
    adwin->delta = cfg->delta;
    adwin->clock = cfg->clock;
    adwin->min_window_length = cfg->min_window_length;
    adwin->grace_period = cfg->grace_period;
    adwin->max_buckets = cfg->max_buckets;
    adwin->last_bucket_row = 0U;
    adwin->drift_detected = false;

    return HT_OK;
}

/* Resets ADWIN window while preserving configuration */
void ht_adwin_reset(ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return;
    }

    ht_float_t delta = adwin->delta;
    uint16_t clock = adwin->clock;
    uint16_t min_window_length = adwin->min_window_length;
    uint16_t grace_period = adwin->grace_period;
    uint8_t max_buckets = adwin->max_buckets;

    memset(adwin->rows, 0, sizeof(adwin->rows));
    adwin->total_sum = HT_CONST(0.0);
    adwin->total_variance = HT_CONST(0.0);
    adwin->width = 0U;
    adwin->mint_time = 0U;
    adwin->n_detections = 0U;
    adwin->last_bucket_row = 0U;
    adwin->drift_detected = false;

    adwin->delta = delta;
    adwin->clock = clock;
    adwin->min_window_length = min_window_length;
    adwin->grace_period = grace_period;
    adwin->max_buckets = max_buckets;
}

/* Analytical cut calculation threshold */
ht_float_t ht_adwin_calculate_cut_epsilon(uint32_t n,
                                          uint32_t n0,
                                          uint32_t n1,
                                          ht_float_t var,
                                          ht_float_t delta,
                                          uint16_t min_w)
{
    if (n <= 1U || delta <= HT_CONST(0.0)) {
        return HT_INFINITY;
    }
    if (n0 < min_w || n1 < min_w) {
        return HT_INFINITY;
    }
    if (var < HT_CONST(0.0)) {
        var = HT_CONST(0.0);
    }

#ifdef HT_USE_FIXED_POINT
    ht_float_t ln_n = ht_adwin_ln_u32(n);
    if (ln_n <= 0) {
        return HT_INFINITY;
    }
    /* dd = ln(2 * ln(n) / delta) = ln(2) + ln(ln(n)) - ln(delta) (overflow-free) */
    ht_float_t dd = HT_SUB(HT_ADD(HT_LN2, HT_LOG(ln_n)), HT_LOG(delta));
    if (dd <= 0 || !HT_ISFINITE(dd)) {
        return HT_INFINITY;
    }

    uint64_t d0 = (uint64_t)(n0 - (uint32_t)min_w + 1U);
    uint64_t d1 = (uint64_t)(n1 - (uint32_t)min_w + 1U);
    /* m = 1/d0 + 1/d1 in Q30 to prevent underflow for large windows up to 81920+ */
    uint64_t m_q30 = ((1ULL << 30) / d0) + ((1ULL << 30) / d1);

    /* term1 = sqrt(2 * m * var * dd) */
    uint64_t dd_q16 = (uint64_t)(uint32_t)dd;
    uint64_t var_q16 = (uint64_t)(uint32_t)var;
    /* m_dd_q30 = (m_q30 * dd_q16) >> 16 */
    uint64_t m_dd_q30 = (m_q30 * dd_q16 + 32768ULL) >> 16;
    /* rad_q16 = (2 * m_dd_q30 * var_q16) >> 30 */
    uint64_t rad_q16;
    if (var_q16 == 0ULL) {
        rad_q16 = 0ULL;
    } else if (m_dd_q30 <= (UINT64_MAX / (2ULL * var_q16 + 1ULL))) {
        rad_q16 = (2ULL * m_dd_q30 * var_q16 + (1ULL << 29)) >> 30;
    } else {
        rad_q16 = 2ULL * ((m_dd_q30 >> 15) * (var_q16 >> 15));
    }
    ht_float_t term1 = (rad_q16 >= (uint64_t)INT32_MAX)
        ? HT_SQRT(INT32_MAX) : HT_SQRT((ht_float_t)rad_q16);

    /* term2 = (2/3) * dd * m in Q16.16: (2 * m_dd_q30) / (3 * 2^14) */
    uint64_t term2_q16 = (2ULL * m_dd_q30 + (3ULL << 13)) / (3ULL << 14);
    ht_float_t term2 = (term2_q16 >= (uint64_t)INT32_MAX) ? INT32_MAX : (ht_float_t)term2_q16;

    return HT_ADD(term1, term2);
#else
    ht_float_t fn = HT_FROM_UINT(n);
    ht_float_t ln_n = HT_LOG(fn);
    if (ln_n <= HT_CONST(0.0)) {
        return HT_INFINITY;
    }
    ht_float_t dd = HT_LOG(HT_DIV(HT_MUL(HT_CONST(2.0), ln_n), delta));
    if (dd <= HT_CONST(0.0)) {
        return HT_INFINITY;
    }
    ht_float_t m = HT_ADD(HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n0 - min_w + 1U)),
                          HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n1 - min_w + 1U)));
    ht_float_t epsilon = HT_ADD(HT_SQRT(HT_MUL(HT_MUL(HT_CONST(2.0), m), HT_MUL(var, dd))),
                                HT_MUL(HT_DIV(HT_CONST(2.0), HT_CONST(3.0)), HT_MUL(dd, m)));
    return epsilon;
#endif
}

/* Compresses buckets across exponential levels */
static void ht_adwin_compress_buckets(ht_adwin_t *adwin) {
    for (uint8_t i = 0; i < HT_ADWIN_LEVELS; ++i) {
        ht_adwin_row_t *row = &adwin->rows[i];
        if (row->count == adwin->max_buckets + 1U) {
            if (i + 1U < HT_ADWIN_LEVELS) {
                ht_adwin_row_t *next_row = &adwin->rows[i + 1U];
                ht_float_t b0_sum = row->sum[0];
                ht_float_t b0_var = row->variance[0];
                ht_float_t b1_sum = row->sum[1];
                ht_float_t b1_var = row->variance[1];

                uint32_t n1 = 1U << i;
                uint32_t n2 = 1U << i;
#ifdef HT_USE_FIXED_POINT
                ht_float_t u1 = ht_adwin_div_u32(b0_sum, n1);
                ht_float_t u2 = ht_adwin_div_u32(b1_sum, n2);
                ht_float_t diff = HT_SUB(u1, u2);
                ht_float_t inc_var = ht_adwin_inc_var_u32(n1, n2, diff);
#else
                ht_float_t u1 = HT_DIV(b0_sum, HT_FROM_UINT(n1));
                ht_float_t u2 = HT_DIV(b1_sum, HT_FROM_UINT(n2));
                ht_float_t diff = HT_SUB(u1, u2);
                ht_float_t inc_var = HT_DIV(HT_MUL(HT_FROM_UINT(n1 * n2), HT_MUL(diff, diff)),
                                            HT_FROM_UINT(n1 + n2));
#endif

                next_row->sum[next_row->count] = HT_ADD(b0_sum, b1_sum);
                next_row->variance[next_row->count] = HT_ADD(HT_ADD(b0_var, b1_var), inc_var);
                next_row->count++;

                if (i + 1U > adwin->last_bucket_row) {
                    adwin->last_bucket_row = (uint8_t)(i + 1U);
                }

                /* Shift remaining buckets left by 2 */
                for (uint8_t k = 2U; k < row->count; ++k) {
                    row->sum[k - 2U] = row->sum[k];
                    row->variance[k - 2U] = row->variance[k];
                }
                row->count -= 2U;
            } else {
                /* Highest level full: drop oldest bucket to prevent overflow */
                uint32_t n_drop = 1U << i;
                ht_float_t b_sum = row->sum[0];
                ht_float_t b_var = row->variance[0];
                adwin->width -= n_drop;
                adwin->total_sum = HT_SUB(adwin->total_sum, b_sum);
                if (adwin->width > 0U) {
#ifdef HT_USE_FIXED_POINT
                    ht_float_t u_drop = ht_adwin_div_u32(b_sum, n_drop);
                    ht_float_t u_rem = ht_adwin_div_u32(adwin->total_sum, adwin->width);
                    ht_float_t diff = HT_SUB(u_drop, u_rem);
                    ht_float_t inc_var = HT_ADD(b_var, ht_adwin_inc_var_u32(n_drop, adwin->width, diff));
#else
                    ht_float_t u_drop = HT_DIV(b_sum, HT_FROM_UINT(n_drop));
                    ht_float_t diff = HT_SUB(u_drop, HT_DIV(adwin->total_sum, HT_FROM_UINT(adwin->width)));
                    ht_float_t inc_var = HT_ADD(b_var,
                        HT_DIV(HT_MUL(HT_MUL(HT_FROM_UINT(n_drop), HT_FROM_UINT(adwin->width)), HT_MUL(diff, diff)),
                               HT_FROM_UINT(n_drop + adwin->width)));
#endif
                    adwin->total_variance = HT_SUB(adwin->total_variance, inc_var);
                    if (adwin->total_variance < HT_CONST(0.0)) {
                        adwin->total_variance = HT_CONST(0.0);
                    }
                } else {
                    adwin->total_variance = HT_CONST(0.0);
                }

                /* Shift remaining buckets left by 1 */
                for (uint8_t k = 1U; k < row->count; ++k) {
                    row->sum[k - 1U] = row->sum[k];
                    row->variance[k - 1U] = row->variance[k];
                }
                row->count -= 1U;
            }
        }
    }
}

/* Deletes the oldest bucket from the tail (highest active level) */
static uint32_t ht_adwin_delete_element(ht_adwin_t *adwin) {
    int32_t last_row = (int32_t)adwin->last_bucket_row;
    while (last_row >= 0 && adwin->rows[last_row].count == 0U) {
        last_row--;
    }
    if (last_row < 0) {
        adwin->last_bucket_row = 0U;
        return 0U;
    }
    adwin->last_bucket_row = (uint8_t)last_row;
    ht_adwin_row_t *row = &adwin->rows[last_row];

    ht_float_t b0_sum = row->sum[0];
    ht_float_t b0_var = row->variance[0];
    uint32_t n1 = 1U << last_row;

    adwin->width -= n1;
    adwin->total_sum = HT_SUB(adwin->total_sum, b0_sum);

    if (adwin->width > 0U) {
#ifdef HT_USE_FIXED_POINT
        ht_float_t u1 = ht_adwin_div_u32(b0_sum, n1);
        ht_float_t u_rem = ht_adwin_div_u32(adwin->total_sum, adwin->width);
        ht_float_t diff = HT_SUB(u1, u_rem);
        ht_float_t inc_var = HT_ADD(b0_var, ht_adwin_inc_var_u32(n1, adwin->width, diff));
#else
        ht_float_t u1 = HT_DIV(b0_sum, HT_FROM_UINT(n1));
        ht_float_t diff = HT_SUB(u1, HT_DIV(adwin->total_sum, HT_FROM_UINT(adwin->width)));
        ht_float_t inc_var = HT_ADD(b0_var,
            HT_DIV(HT_MUL(HT_MUL(HT_FROM_UINT(n1), HT_FROM_UINT(adwin->width)), HT_MUL(diff, diff)),
                   HT_FROM_UINT(n1 + adwin->width)));
#endif
        adwin->total_variance = HT_SUB(adwin->total_variance, inc_var);
        if (adwin->total_variance < HT_CONST(0.0)) {
            adwin->total_variance = HT_CONST(0.0);
        }
    } else {
        adwin->total_variance = HT_CONST(0.0);
    }

    /* Shift remaining buckets in this row left by 1 */
    for (uint8_t k = 1U; k < row->count; ++k) {
        row->sum[k - 1U] = row->sum[k];
        row->variance[k - 1U] = row->variance[k];
    }
    row->count -= 1U;

    while (last_row > 0 && adwin->rows[last_row].count == 0U) {
        last_row--;
    }
    adwin->last_bucket_row = (uint8_t)last_row;

    return n1;
}

/* Inserts a single observation into ADWIN */
static void ht_adwin_insert_element(ht_adwin_t *adwin, ht_float_t value) {
#ifdef HT_USE_FIXED_POINT
    /* Prevent Q16.16 total_sum saturation by evicting oldest buckets (native sliding window) */
    const int64_t sum_cap = (int64_t)HT_CONST(28672.0);
    while (adwin->width > 0U) {
        int64_t next_sum = (int64_t)adwin->total_sum + (int64_t)value;
        int64_t next_abs = (next_sum < 0) ? -next_sum : next_sum;
        if (next_abs < sum_cap) {
            break;
        }
        if (ht_adwin_delete_element(adwin) == 0U) {
            break;
        }
    }
#endif
    adwin->width++;
    adwin->mint_time++;

    /* Insert new bucket in row 0 */
    ht_adwin_row_t *row0 = &adwin->rows[0];
    row0->sum[row0->count] = value;
    row0->variance[row0->count] = HT_CONST(0.0);
    row0->count++;

    if (adwin->width > 1U) {
#ifdef HT_USE_FIXED_POINT
        ht_float_t prev_mean = ht_adwin_div_u32(adwin->total_sum, adwin->width - 1U);
        ht_float_t diff = HT_SUB(value, prev_mean);
        ht_float_t inc_var = ht_adwin_inc_var_u32(adwin->width - 1U, 1U, diff);
#else
        ht_float_t prev_w = HT_FROM_UINT(adwin->width - 1U);
        ht_float_t curr_w = HT_FROM_UINT(adwin->width);
        ht_float_t prev_mean = HT_DIV(adwin->total_sum, prev_w);
        ht_float_t diff = HT_SUB(value, prev_mean);
        ht_float_t inc_var = HT_DIV(HT_MUL(prev_w, HT_MUL(diff, diff)), curr_w);
#endif
        adwin->total_variance = HT_ADD(adwin->total_variance, inc_var);
    }
    adwin->total_sum = HT_ADD(adwin->total_sum, value);

    ht_adwin_compress_buckets(adwin);
}

/* Updates ADWIN with a stream observation */
bool ht_adwin_update(ht_adwin_t *adwin, ht_float_t value) {
    if (adwin == NULL || HT_ISNAN(value) || !HT_ISFINITE(value)) {
        return false;
    }

    ht_adwin_insert_element(adwin, value);
    bool drift = false;

    if ((adwin->mint_time % adwin->clock == 0U) && (adwin->width > adwin->grace_period)) {
        bool reduce_width = true;
        while (reduce_width) {
            reduce_width = false;
            bool exit_check = false;
            uint32_t n0 = 0U;
            uint32_t n1 = adwin->width;
            ht_float_t u0 = HT_CONST(0.0);
            ht_float_t u1 = adwin->total_sum;

            int32_t row_idx = (int32_t)adwin->last_bucket_row;
            while (row_idx >= 0 && !exit_check) {
                const ht_adwin_row_t *row = &adwin->rows[row_idx];
                for (uint8_t k = 0; k < row->count; ++k) {
                    uint32_t n2 = 1U << row_idx;
                    n0 += n2;
                    n1 -= n2;
                    u0 = HT_ADD(u0, row->sum[k]);
                    u1 = HT_SUB(u1, row->sum[k]);

                    if (n0 >= adwin->min_window_length && n1 >= adwin->min_window_length) {
#ifdef HT_USE_FIXED_POINT
                        ht_float_t mean0 = ht_adwin_div_u32(u0, n0);
                        ht_float_t mean1 = ht_adwin_div_u32(u1, n1);
                        ht_float_t var_in_win = ht_adwin_div_u32(adwin->total_variance, adwin->width);
#else
                        ht_float_t mean0 = HT_DIV(u0, HT_FROM_UINT(n0));
                        ht_float_t mean1 = HT_DIV(u1, HT_FROM_UINT(n1));
                        ht_float_t var_in_win = HT_DIV(adwin->total_variance, HT_FROM_UINT(adwin->width));
#endif
                        ht_float_t absval = HT_FABS(HT_SUB(mean0, mean1));
                        ht_float_t epsilon = ht_adwin_calculate_cut_epsilon(
                            adwin->width, n0, n1, var_in_win, adwin->delta, adwin->min_window_length
                        );

                        if (absval > epsilon) {
                            drift = true;
                            reduce_width = true;
                            ht_adwin_delete_element(adwin);
                            exit_check = true;
                            break;
                        }
                    }
                }
                row_idx--;
            }
        }
    }

    if (drift) {
        adwin->n_detections++;
    }
    adwin->drift_detected = drift;
    return drift;
}

/* Getters */
ht_float_t ht_adwin_get_estimation(const ht_adwin_t *adwin) {
    if (adwin == NULL || adwin->width == 0U) {
        return HT_CONST(0.0);
    }
#ifdef HT_USE_FIXED_POINT
    return ht_adwin_div_u32(adwin->total_sum, adwin->width);
#else
    return HT_DIV(adwin->total_sum, HT_FROM_UINT(adwin->width));
#endif
}

ht_float_t ht_adwin_get_variance(const ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return HT_CONST(0.0);
    }
    return adwin->total_variance;
}

ht_float_t ht_adwin_get_variance_in_window(const ht_adwin_t *adwin) {
    if (adwin == NULL || adwin->width == 0U) {
        return HT_CONST(0.0);
    }
#ifdef HT_USE_FIXED_POINT
    return ht_adwin_div_u32(adwin->total_variance, adwin->width);
#else
    return HT_DIV(adwin->total_variance, HT_FROM_UINT(adwin->width));
#endif
}

uint32_t ht_adwin_get_width(const ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return 0U;
    }
    return adwin->width;
}

ht_float_t ht_adwin_get_total(const ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return HT_CONST(0.0);
    }
    return adwin->total_sum;
}

uint32_t ht_adwin_get_n_detections(const ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return 0U;
    }
    return adwin->n_detections;
}

/* --- ADWIN Pool Allocator Implementation --- */

ht_status_t ht_adwin_pool_init(ht_adwin_pool_t *pool) {
    if (pool == NULL) {
        return HT_ERR_INVALID_ARG;
    }

    memset(pool, 0, sizeof(*pool));
    for (uint16_t i = 0; i < HT_MAX_ADWIN_DETECTORS; ++i) {
        pool->free_indices[i] = (uint16_t)(HT_MAX_ADWIN_DETECTORS - 1U - i);
    }
    pool->free_top = HT_MAX_ADWIN_DETECTORS;
    pool->allocated_count = 0U;

    return HT_OK;
}

uint16_t ht_adwin_pool_alloc(ht_adwin_pool_t *pool, const ht_adwin_config_t *cfg) {
    if (pool == NULL || pool->free_top == 0U) {
        return HT_INVALID_IDX;
    }

    pool->free_top--;
    uint16_t idx = pool->free_indices[pool->free_top];

    uint16_t word_idx = idx / 32U;
    uint32_t bit_mask = 1U << (idx % 32U);
    pool->in_use_mask[word_idx] |= bit_mask;
    pool->allocated_count++;

    ht_adwin_init(&pool->pool[idx], cfg);
    return idx;
}

ht_status_t ht_adwin_pool_free(ht_adwin_pool_t *pool, uint16_t idx) {
    if (pool == NULL || idx >= HT_MAX_ADWIN_DETECTORS) {
        return HT_ERR_INVALID_ARG;
    }

    uint16_t word_idx = idx / 32U;
    uint32_t bit_mask = 1U << (idx % 32U);
    if ((pool->in_use_mask[word_idx] & bit_mask) == 0U) {
        return HT_ERR_INVALID_ARG;
    }

    pool->in_use_mask[word_idx] &= ~bit_mask;
    pool->free_indices[pool->free_top] = idx;
    pool->free_top++;
    pool->allocated_count--;

    return HT_OK;
}

ht_adwin_t *ht_adwin_pool_get(ht_adwin_pool_t *pool, uint16_t idx) {
    if (pool == NULL || idx >= HT_MAX_ADWIN_DETECTORS) {
        return NULL;
    }

    uint16_t word_idx = idx / 32U;
    uint32_t bit_mask = 1U << (idx % 32U);
    if ((pool->in_use_mask[word_idx] & bit_mask) == 0U) {
        return NULL;
    }

    return &pool->pool[idx];
}

void ht_adwin_pool_reset(ht_adwin_pool_t *pool) {
    if (pool == NULL) {
        return;
    }
    ht_adwin_pool_init(pool);
}

size_t ht_adwin_pool_bytes_used(const ht_adwin_pool_t *pool) {
    if (pool == NULL) {
        return 0U;
    }
    return sizeof(ht_adwin_pool_t);
}

/* --- HAT Concept Drift Helpers Implementation --- */

ht_float_t ht_hat_calculate_switch_bound(ht_float_t e_old,
                                         uint32_t n_old,
                                         uint32_t n_alt,
                                         ht_float_t alpha)
{
    if (n_old == 0U || n_alt == 0U || alpha <= HT_CONST(0.0) || alpha >= HT_CONST(1.0)) {
        return HT_INFINITY;
    }

    ht_float_t var = HT_MUL(e_old, HT_SUB(HT_CONST(1.0), e_old));
    if (var < HT_CONST(0.0)) {
        var = HT_CONST(0.0);
    }

#ifdef HT_USE_FIXED_POINT
    ht_float_t ln_term = HT_SUB(HT_LN2, HT_LOG(alpha));
    if (ln_term <= 0) {
        return 0;
    }
    uint64_t inv_n_q30 = ((1ULL << 30) / (uint64_t)n_alt) + ((1ULL << 30) / (uint64_t)n_old);
    uint64_t v_ln_q16 = ((uint64_t)(uint32_t)var * (uint64_t)(uint32_t)ln_term + 32768ULL) >> 16;
    uint64_t rad_q16 = (2ULL * v_ln_q16 * inv_n_q30 + (1ULL << 29)) >> 30;
    return (rad_q16 >= (uint64_t)INT32_MAX) ? HT_SQRT(INT32_MAX) : HT_SQRT((ht_float_t)rad_q16);
#else
    ht_float_t ln_term = HT_LOG(HT_DIV(HT_CONST(2.0), alpha));
    ht_float_t inv_n = HT_ADD(HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_alt)),
                              HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_old)));

    return HT_SQRT(HT_MUL(HT_MUL(HT_CONST(2.0), var), HT_MUL(ln_term, inv_n)));
#endif
}

ht_hat_action_t ht_hat_evaluate_alternate(ht_float_t e_old,
                                          uint32_t n_old,
                                          ht_float_t e_alt,
                                          uint32_t n_alt,
                                          uint32_t min_obs,
                                          ht_float_t alpha)
{
    if (n_old <= min_obs || n_alt <= min_obs) {
        return HT_HAT_ACTION_NONE;
    }

    ht_float_t bound = ht_hat_calculate_switch_bound(e_old, n_old, n_alt, alpha);
    if (bound == HT_INFINITY) {
        return HT_HAT_ACTION_NONE;
    }

    if (bound < HT_SUB(e_old, e_alt)) {
        return HT_HAT_ACTION_SWITCH;
    }
    if (bound < HT_SUB(e_alt, e_old)) {
        return HT_HAT_ACTION_PRUNE;
    }

    return HT_HAT_ACTION_NONE;
}

bool ht_hat_check_error_drift(ht_adwin_t *adwin,
                              ht_float_t *mean_error,
                              uint32_t *error_count,
                              uint8_t detec_in)
{
    if (adwin == NULL || mean_error == NULL || error_count == NULL) {
        return false;
    }

    ht_float_t old_err = *mean_error;
    ht_float_t det_val = HT_FROM_UINT(detec_in);
    bool drift = ht_adwin_update(adwin, det_val);

    /* Update incremental mean: mean += (val - mean) / n */
    (*error_count)++;
#ifdef HT_USE_FIXED_POINT
    *mean_error = HT_ADD(*mean_error, ht_adwin_div_u32(HT_SUB(det_val, *mean_error), *error_count));
#else
    *mean_error = HT_ADD(*mean_error, HT_DIV(HT_SUB(det_val, *mean_error), HT_FROM_UINT(*error_count)));
#endif

    if (drift) {
        if (*mean_error <= old_err) {
            *mean_error = HT_CONST(0.0);
            *error_count = 0U;
            return false;
        }
        *mean_error = HT_CONST(0.0);
        *error_count = 0U;
        return true;
    }

    return false;
}

void ht_hat_combine_leaf_predictions(const ht_float_t *leaf_probas,
                                     const ht_float_t *mean_errors,
                                     uint16_t n_leaves,
                                     uint8_t n_classes,
                                     ht_float_t out_proba[HT_MAX_CLASSES])
{
    if (out_proba == NULL || n_classes == 0U) {
        return;
    }

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        out_proba[c] = HT_CONST(0.0);
    }

    if (leaf_probas == NULL || mean_errors == NULL || n_leaves == 0U) {
        ht_float_t uniform = HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_classes));
        for (uint8_t c = 0; c < n_classes; ++c) {
            out_proba[c] = uniform;
        }
        return;
    }

    if (n_leaves == 1U) {
        ht_float_t sum = HT_CONST(0.0);
        for (uint8_t c = 0; c < n_classes; ++c) {
            sum = HT_ADD(sum, leaf_probas[c]);
        }
        if (sum > HT_CONST(0.0)) {
            for (uint8_t c = 0; c < n_classes; ++c) {
                out_proba[c] = HT_DIV(leaf_probas[c], sum);
            }
        } else {
            ht_float_t uniform = HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_classes));
            for (uint8_t c = 0; c < n_classes; ++c) {
                out_proba[c] = uniform;
            }
        }
        return;
    }

#ifdef HT_USE_FIXED_POINT
    int64_t acc_proba[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        acc_proba[c] = 0;
    }
    int64_t total_weight64 = 0;

    for (uint16_t i = 0; i < n_leaves; ++i) {
        ht_float_t err = mean_errors[i];
        int64_t w64;
        if (err <= HT_CONST(0.015625)) {
            w64 = (int64_t)HT_CONST(4096.0);
        } else {
            ht_float_t e2 = HT_MUL(err, err);
            w64 = (e2 > 0) ? (int64_t)HT_DIV(HT_FIXED_ONE, e2) : (int64_t)HT_CONST(4096.0);
            if (w64 > (int64_t)HT_CONST(4096.0)) {
                w64 = (int64_t)HT_CONST(4096.0);
            }
        }

        const ht_float_t *dist = &leaf_probas[(size_t)i * HT_MAX_CLASSES];
        int64_t leaf_sum64 = 0;
        for (uint8_t c = 0; c < n_classes; ++c) {
            if (dist[c] > 0) {
                leaf_sum64 += (int64_t)dist[c];
            }
        }

        if (leaf_sum64 > 0) {
            for (uint8_t c = 0; c < n_classes; ++c) {
                int64_t p_norm = (((int64_t)dist[c]) << 16) / leaf_sum64;
                int64_t contrib = (w64 * p_norm + 32768LL) >> 16;
                acc_proba[c] += contrib;
                total_weight64 += contrib;
            }
        }
    }

    if (total_weight64 > 0) {
        for (uint8_t c = 0; c < n_classes; ++c) {
            uint64_t num = ((uint64_t)acc_proba[c] << 16) + ((uint64_t)total_weight64 >> 1);
            out_proba[c] = ht_fixed_saturate64((int64_t)(num / (uint64_t)total_weight64));
        }
    } else {
        ht_float_t uniform = HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_classes));
        for (uint8_t c = 0; c < n_classes; ++c) {
            out_proba[c] = uniform;
        }
    }
#else
    ht_float_t total_weight = HT_CONST(0.0);
    for (uint16_t i = 0; i < n_leaves; ++i) {
        ht_float_t err = mean_errors[i];
        ht_float_t w;
        if (err <= HT_CONST(1e-6)) {
            w = HT_CONST(1e12);
        } else {
            w = HT_DIV(HT_CONST(1.0), HT_MUL(err, err));
        }

        const ht_float_t *dist = &leaf_probas[(size_t)i * HT_MAX_CLASSES];
        ht_float_t leaf_sum = HT_CONST(0.0);
        for (uint8_t c = 0; c < n_classes; ++c) {
            leaf_sum = HT_ADD(leaf_sum, dist[c]);
        }

        if (leaf_sum > HT_CONST(0.0)) {
            for (uint8_t c = 0; c < n_classes; ++c) {
                ht_float_t p_norm = HT_DIV(dist[c], leaf_sum);
                ht_float_t contrib = HT_MUL(w, p_norm);
                out_proba[c] = HT_ADD(out_proba[c], contrib);
                total_weight = HT_ADD(total_weight, contrib);
            }
        }
    }

    if (total_weight > HT_CONST(0.0)) {
        for (uint8_t c = 0; c < n_classes; ++c) {
            out_proba[c] = HT_DIV(out_proba[c], total_weight);
        }
    } else {
        ht_float_t uniform = HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_classes));
        for (uint8_t c = 0; c < n_classes; ++c) {
            out_proba[c] = uniform;
        }
    }
#endif
}

bool ht_hat_check_reg_error_drift(ht_adwin_t *adwin,
                                  ht_float_t *mean_error,
                                  ht_float_t *error_m2,
                                  uint32_t *error_count,
                                  ht_float_t abs_error) {
    if (adwin == NULL || mean_error == NULL || error_m2 == NULL || error_count == NULL) {
        return false;
    }

    ht_float_t old_error = *mean_error;

    bool error_change = ht_adwin_update(adwin, abs_error);

    /* Update Welford accumulators for absolute error */
    (*error_count)++;
    uint32_t n = *error_count;
    ht_float_t delta = HT_SUB(abs_error, *mean_error);
#ifdef HT_USE_FIXED_POINT
    *mean_error = HT_ADD(*mean_error, ht_adwin_div_u32(delta, n));
    ht_float_t delta2 = HT_SUB(abs_error, *mean_error);
    *error_m2 = HT_ADD(*error_m2, HT_MUL(delta, delta2));
#else
    *mean_error = HT_ADD(*mean_error, HT_DIV(delta, HT_FROM_UINT(n)));
    ht_float_t delta2 = HT_SUB(abs_error, *mean_error);
    *error_m2 = HT_ADD(*error_m2, HT_MUL(delta, delta2));
#endif

    if (error_change && (*mean_error <= old_error)) {
        *mean_error = HT_CONST(0.0);
        *error_m2 = HT_CONST(0.0);
        *error_count = 0U;
        return false;
    }

    if (error_change) {
        *mean_error = HT_CONST(0.0);
        *error_m2 = HT_CONST(0.0);
        *error_count = 0U;
        return true;
    }

    return false;
}

ht_hat_action_t ht_hat_evaluate_reg_alternate(ht_float_t e_old,
                                              ht_float_t var_old,
                                              uint32_t n_old,
                                              ht_float_t e_alt,
                                              ht_float_t var_alt,
                                              uint32_t n_alt,
                                              uint32_t min_obs,
                                              ht_float_t alpha) {
    if (n_old <= min_obs || n_alt <= min_obs) {
        return HT_HAT_ACTION_NONE;
    }

#ifdef HT_USE_FIXED_POINT
    ht_float_t se2 = HT_ADD(ht_adwin_div_u32(var_alt, n_alt), ht_adwin_div_u32(var_old, n_old));
#else
    ht_float_t se2 = HT_ADD(HT_DIV(var_alt, HT_FROM_UINT(n_alt)),
                            HT_DIV(var_old, HT_FROM_UINT(n_old)));
#endif
    ht_float_t diff = HT_SUB(e_alt, e_old);
    ht_float_t p_value;

    if (se2 <= HT_CONST(0.0)) {
        if (diff == HT_CONST(0.0)) {
            return HT_HAT_ACTION_NONE;
        }
        p_value = HT_CONST(0.0);
    } else {
        ht_float_t se = HT_SQRT(se2);
        if (se <= HT_CONST(0.0)) {
            p_value = HT_CONST(0.0);
        } else {
            ht_float_t z = HT_DIV(diff, se);
            ht_float_t abs_z = HT_FABS(z);
            p_value = HT_ERFC(HT_DIV(abs_z, HT_SQRT2));
        }
    }

    if (p_value <= alpha) {
        if (e_alt < e_old) {
            return HT_HAT_ACTION_SWITCH;
        } else {
            return HT_HAT_ACTION_PRUNE;
        }
    }

    return HT_HAT_ACTION_NONE;
}
