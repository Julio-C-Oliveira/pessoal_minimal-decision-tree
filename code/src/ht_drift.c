/**
 * @file ht_drift.c
 * @brief Implementation of the ADWIN Concept Drift Detector and Pool Allocator.
 */

#include "ht_drift.h"

/* Populates default ADWIN configuration matching River defaults */
void ht_adwin_default_config(ht_adwin_config_t *cfg) {
    if (cfg == NULL) {
        return;
    }
    cfg->delta = (ht_float_t)0.002;
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
        if (cfg->delta <= (ht_float_t)0.0 || cfg->delta >= (ht_float_t)1.0 ||
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
    adwin->total_sum = (ht_float_t)0.0;
    adwin->total_variance = (ht_float_t)0.0;
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
    if (n <= 1U || delta <= (ht_float_t)0.0) {
        return HT_INFINITY;
    }
    ht_float_t fn = (ht_float_t)n;
    ht_float_t ln_n = HT_LOG(fn);
    if (ln_n <= (ht_float_t)0.0) {
        return HT_INFINITY;
    }
    ht_float_t dd = HT_LOG(((ht_float_t)2.0 * ln_n) / delta);
    if (dd <= (ht_float_t)0.0) {
        return HT_INFINITY;
    }
    if (var < (ht_float_t)0.0) {
        var = (ht_float_t)0.0;
    }
    if (n0 < min_w || n1 < min_w) {
        return HT_INFINITY;
    }
    ht_float_t m = ((ht_float_t)1.0 / (ht_float_t)(n0 - min_w + 1U)) +
                   ((ht_float_t)1.0 / (ht_float_t)(n1 - min_w + 1U));
    ht_float_t epsilon = HT_SQRT((ht_float_t)2.0 * m * var * dd) +
                         ((ht_float_t)2.0 / (ht_float_t)3.0) * dd * m;
    return epsilon;
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
                ht_float_t u1 = b0_sum / (ht_float_t)n1;
                ht_float_t u2 = b1_sum / (ht_float_t)n2;
                ht_float_t diff = u1 - u2;
                ht_float_t inc_var = ((ht_float_t)(n1 * n2) * (diff * diff)) / (ht_float_t)(n1 + n2);

                next_row->sum[next_row->count] = b0_sum + b1_sum;
                next_row->variance[next_row->count] = b0_var + b1_var + inc_var;
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
                adwin->total_sum -= b_sum;
                if (adwin->width > 0U) {
                    ht_float_t u_drop = b_sum / (ht_float_t)n_drop;
                    ht_float_t diff = u_drop - (adwin->total_sum / (ht_float_t)adwin->width);
                    ht_float_t inc_var = b_var + ((ht_float_t)n_drop * (ht_float_t)adwin->width * (diff * diff)) /
                                                 (ht_float_t)(n_drop + adwin->width);
                    adwin->total_variance -= inc_var;
                    if (adwin->total_variance < (ht_float_t)0.0) {
                        adwin->total_variance = (ht_float_t)0.0;
                    }
                } else {
                    adwin->total_variance = (ht_float_t)0.0;
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
    adwin->total_sum -= b0_sum;
    ht_float_t u1 = b0_sum / (ht_float_t)n1;

    if (adwin->width > 0U) {
        ht_float_t diff = u1 - (adwin->total_sum / (ht_float_t)adwin->width);
        ht_float_t inc_var = b0_var + ((ht_float_t)n1 * (ht_float_t)adwin->width * (diff * diff)) /
                                      (ht_float_t)(n1 + adwin->width);
        adwin->total_variance -= inc_var;
        if (adwin->total_variance < (ht_float_t)0.0) {
            adwin->total_variance = (ht_float_t)0.0;
        }
    } else {
        adwin->total_variance = (ht_float_t)0.0;
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
    adwin->width++;
    adwin->mint_time++;

    /* Insert new bucket in row 0 */
    ht_adwin_row_t *row0 = &adwin->rows[0];
    row0->sum[row0->count] = value;
    row0->variance[row0->count] = (ht_float_t)0.0;
    row0->count++;

    if (adwin->width > 1U) {
        ht_float_t prev_w = (ht_float_t)(adwin->width - 1U);
        ht_float_t curr_w = (ht_float_t)adwin->width;
        ht_float_t prev_mean = adwin->total_sum / prev_w;
        ht_float_t diff = value - prev_mean;
        ht_float_t inc_var = (prev_w * (diff * diff)) / curr_w;
        adwin->total_variance += inc_var;
    }
    adwin->total_sum += value;

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
            ht_float_t u0 = (ht_float_t)0.0;
            ht_float_t u1 = adwin->total_sum;

            int32_t row_idx = (int32_t)adwin->last_bucket_row;
            while (row_idx >= 0 && !exit_check) {
                const ht_adwin_row_t *row = &adwin->rows[row_idx];
                for (uint8_t k = 0; k < row->count; ++k) {
                    uint32_t n2 = 1U << row_idx;
                    n0 += n2;
                    n1 -= n2;
                    u0 += row->sum[k];
                    u1 -= row->sum[k];

                    if (n0 >= adwin->min_window_length && n1 >= adwin->min_window_length) {
                        ht_float_t mean0 = u0 / (ht_float_t)n0;
                        ht_float_t mean1 = u1 / (ht_float_t)n1;
                        ht_float_t absval = HT_FABS(mean0 - mean1);
                        ht_float_t var_in_win = adwin->total_variance / (ht_float_t)adwin->width;
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
        return (ht_float_t)0.0;
    }
    return adwin->total_sum / (ht_float_t)adwin->width;
}

ht_float_t ht_adwin_get_variance(const ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return (ht_float_t)0.0;
    }
    return adwin->total_variance;
}

ht_float_t ht_adwin_get_variance_in_window(const ht_adwin_t *adwin) {
    if (adwin == NULL || adwin->width == 0U) {
        return (ht_float_t)0.0;
    }
    return adwin->total_variance / (ht_float_t)adwin->width;
}

uint32_t ht_adwin_get_width(const ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return 0U;
    }
    return adwin->width;
}

ht_float_t ht_adwin_get_total(const ht_adwin_t *adwin) {
    if (adwin == NULL) {
        return (ht_float_t)0.0;
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
        /* free list stack: pop from top gives index 0 first */
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
        /* Double-free protection */
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
    if (n_old == 0U || n_alt == 0U || alpha <= (ht_float_t)0.0 || alpha >= (ht_float_t)1.0) {
        return HT_INFINITY;
    }

    ht_float_t var = e_old * ((ht_float_t)1.0 - e_old);
    if (var < (ht_float_t)0.0) {
        var = (ht_float_t)0.0;
    }

    ht_float_t ln_term = HT_LOG(((ht_float_t)2.0) / alpha);
    ht_float_t inv_n = ((ht_float_t)1.0 / (ht_float_t)n_alt) + ((ht_float_t)1.0 / (ht_float_t)n_old);

    return HT_SQRT((ht_float_t)2.0 * var * ln_term * inv_n);
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

    if (bound < (e_old - e_alt)) {
        return HT_HAT_ACTION_SWITCH;
    }
    if (bound < (e_alt - e_old)) {
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
    bool drift = ht_adwin_update(adwin, (ht_float_t)detec_in);

    /* Update incremental mean: mean += (val - mean) / n */
    (*error_count)++;
    *mean_error += ((ht_float_t)detec_in - *mean_error) / (ht_float_t)(*error_count);

    if (drift) {
        if (*mean_error <= old_err) {
            /* Error is decreasing or unchanged: false alarm for drift adaptation, reset estimator */
            *mean_error = (ht_float_t)0.0;
            *error_count = 0U;
            return false;
        }
        /* Error is strictly increasing: genuine drift detected */
        *mean_error = (ht_float_t)0.0;
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
        out_proba[c] = (ht_float_t)0.0;
    }

    if (leaf_probas == NULL || mean_errors == NULL || n_leaves == 0U) {
        ht_float_t uniform = (ht_float_t)1.0 / (ht_float_t)n_classes;
        for (uint8_t c = 0; c < n_classes; ++c) {
            out_proba[c] = uniform;
        }
        return;
    }

    if (n_leaves == 1U) {
        ht_float_t sum = (ht_float_t)0.0;
        for (uint8_t c = 0; c < n_classes; ++c) {
            sum += leaf_probas[c];
        }
        if (sum > (ht_float_t)0.0) {
            for (uint8_t c = 0; c < n_classes; ++c) {
                out_proba[c] = leaf_probas[c] / sum;
            }
        } else {
            ht_float_t uniform = (ht_float_t)1.0 / (ht_float_t)n_classes;
            for (uint8_t c = 0; c < n_classes; ++c) {
                out_proba[c] = uniform;
            }
        }
        return;
    }

    ht_float_t total_weight = (ht_float_t)0.0;
    for (uint16_t i = 0; i < n_leaves; ++i) {
        ht_float_t err = mean_errors[i];
        ht_float_t w;
        if (err <= (ht_float_t)1e-6) {
            w = (ht_float_t)1e12;
        } else {
            w = ((ht_float_t)1.0) / (err * err);
        }

        const ht_float_t *dist = &leaf_probas[(size_t)i * HT_MAX_CLASSES];
        ht_float_t leaf_sum = (ht_float_t)0.0;
        for (uint8_t c = 0; c < n_classes; ++c) {
            leaf_sum += dist[c];
        }

        if (leaf_sum > (ht_float_t)0.0) {
            for (uint8_t c = 0; c < n_classes; ++c) {
                ht_float_t p_norm = dist[c] / leaf_sum;
                out_proba[c] += w * p_norm;
                total_weight += w * p_norm;
            }
        }
    }

    if (total_weight > (ht_float_t)0.0) {
        for (uint8_t c = 0; c < n_classes; ++c) {
            out_proba[c] /= total_weight;
        }
    } else {
        ht_float_t uniform = (ht_float_t)1.0 / (ht_float_t)n_classes;
        for (uint8_t c = 0; c < n_classes; ++c) {
            out_proba[c] = uniform;
        }
    }
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
    ht_float_t delta = abs_error - *mean_error;
    *mean_error += delta / (ht_float_t)n;
    ht_float_t delta2 = abs_error - *mean_error;
    *error_m2 += delta * delta2;

    /* Unilateral error filter: if error is decreasing or unchanged, filter out change */
    if (error_change && (*mean_error <= old_error)) {
        /* Reset error estimator */
        *mean_error = (ht_float_t)0.0;
        *error_m2 = (ht_float_t)0.0;
        *error_count = 0U;
        return false;
    }

    if (error_change) {
        /* Genuine drift: reset error estimator and signal drift */
        *mean_error = (ht_float_t)0.0;
        *error_m2 = (ht_float_t)0.0;
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

    ht_float_t se2 = (var_alt / (ht_float_t)n_alt) + (var_old / (ht_float_t)n_old);
    ht_float_t diff = e_alt - e_old;
    ht_float_t p_value;

    if (se2 <= (ht_float_t)0.0) {
        if (diff == (ht_float_t)0.0) {
            return HT_HAT_ACTION_NONE;
        }
        p_value = (ht_float_t)0.0;
    } else {
        ht_float_t z = diff / HT_SQRT(se2);
        ht_float_t abs_z = HT_FABS(z);
        p_value = HT_ERFC(abs_z / HT_SQRT2);
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
