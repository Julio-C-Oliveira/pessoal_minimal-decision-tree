/**
 * @file ht_prep.c
 * @brief Implementation of streaming data preprocessing micro-pipelines (libht_prep).
 */

#include "ht_prep.h"
#include <string.h>

/* ========================================================================= */
/* --- 1. Online Scaler Implementation ------------------------------------- */
/* ========================================================================= */

void ht_scaler_init(ht_scaler_t *scaler) {
    if (scaler == NULL) return;
    memset(scaler, 0, sizeof(*scaler));
    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        scaler->stats[f].min_val = HT_INFINITY;
        scaler->stats[f].max_val = -HT_INFINITY;
        scaler->mode[f] = HT_SCALER_NONE;
    }
}

void ht_scaler_set_mode(ht_scaler_t *scaler, uint8_t feat_idx, ht_scaler_type_t mode) {
    if (scaler == NULL || feat_idx >= HT_MAX_FEATURES) return;
    scaler->mode[feat_idx] = mode;
}

void ht_scaler_update_one(ht_scaler_t *scaler, uint8_t feat_idx, ht_float_t val, ht_float_t weight) {
    if (scaler == NULL || feat_idx >= HT_MAX_FEATURES || HT_ISNAN(val) || weight <= (ht_float_t)0.0) {
        return;
    }
    ht_scaler_stat_t *s = &scaler->stats[feat_idx];
    if (s->count <= (ht_float_t)0.0) {
        s->count = weight;
        s->mean = val;
        s->m2 = (ht_float_t)0.0;
        s->min_val = val;
        s->max_val = val;
    } else {
        ht_float_t new_count = s->count + weight;
        ht_float_t delta1 = val - s->mean;
        s->mean += (weight / new_count) * delta1;
        ht_float_t delta2 = val - s->mean;
        s->m2 += weight * delta1 * delta2;
        s->count = new_count;
        if (val < s->min_val) s->min_val = val;
        if (val > s->max_val) s->max_val = val;
    }
}

ht_float_t ht_scaler_transform_one(const ht_scaler_t *scaler, uint8_t feat_idx, ht_float_t val) {
    if (scaler == NULL || feat_idx >= HT_MAX_FEATURES || HT_ISNAN(val)) {
        return val;
    }
    ht_scaler_type_t m = scaler->mode[feat_idx];
    if (m == HT_SCALER_STANDARD) {
        const ht_scaler_stat_t *s = &scaler->stats[feat_idx];
        if (s->count <= (ht_float_t)1.0) {
            return (ht_float_t)0.0;
        }
        ht_float_t var = (s->m2 > (ht_float_t)0.0) ? (s->m2 / (s->count - (ht_float_t)1.0)) : (ht_float_t)0.0;
        ht_float_t stddev = HT_SQRT(var + (ht_float_t)1e-6);
        return (val - s->mean) / stddev;
    } else if (m == HT_SCALER_MINMAX) {
        const ht_scaler_stat_t *s = &scaler->stats[feat_idx];
        ht_float_t range = s->max_val - s->min_val;
        if (range <= (ht_float_t)1e-6) {
            return (ht_float_t)0.0;
        }
        ht_float_t z = (val - s->min_val) / range;
        if (z < (ht_float_t)0.0) z = (ht_float_t)0.0;
        if (z > (ht_float_t)1.0) z = (ht_float_t)1.0;
        return z;
    }
    return val;
}

ht_float_t ht_scaler_learn_transform_one(ht_scaler_t *scaler, uint8_t feat_idx, ht_float_t val, ht_float_t weight) {
    ht_scaler_update_one(scaler, feat_idx, val, weight);
    return ht_scaler_transform_one(scaler, feat_idx, val);
}

/* ========================================================================= */
/* --- 2. Hampel Moving Median Filter Implementation ----------------------- */
/* ========================================================================= */

void ht_hampel_init(ht_filter_hampel_t *f) {
    if (f == NULL) return;
    memset(f, 0, sizeof(*f));
    f->n_sigma = (ht_float_t)3.0;
}

void ht_hampel_set_enabled(ht_filter_hampel_t *f, uint8_t feat_idx, bool enabled) {
    if (f == NULL || feat_idx >= HT_MAX_FEATURES) return;
    f->enabled[feat_idx] = enabled;
}

void ht_hampel_set_threshold(ht_filter_hampel_t *f, ht_float_t n_sigma) {
    if (f == NULL || n_sigma <= (ht_float_t)0.0) return;
    f->n_sigma = n_sigma;
}

ht_float_t ht_hampel_filter_one(ht_filter_hampel_t *f, uint8_t feat_idx, ht_float_t val) {
    if (f == NULL || feat_idx >= HT_MAX_FEATURES || !f->enabled[feat_idx] || HT_ISNAN(val)) {
        return val;
    }

    uint8_t idx = f->head[feat_idx];
    f->window[feat_idx][idx] = val;
    f->head[feat_idx] = (uint8_t)((idx + 1U) % HT_HAMPEL_WINDOW_SIZE);
    if (f->count[feat_idx] < HT_HAMPEL_WINDOW_SIZE) {
        f->count[feat_idx]++;
    }

    uint8_t n = f->count[feat_idx];
    if (n < 3U) {
        return val;
    }

    /* Stack array for sorting window values */
    ht_float_t sorted[HT_HAMPEL_WINDOW_SIZE];
    for (uint8_t i = 0; i < n; ++i) {
        sorted[i] = f->window[feat_idx][i];
    }
    /* Insertion sort */
    for (uint8_t i = 1; i < n; ++i) {
        ht_float_t key = sorted[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    ht_float_t med;
    if ((n % 2U) == 1U) {
        med = sorted[n / 2U];
    } else {
        med = (sorted[n / 2U - 1U] + sorted[n / 2U]) * (ht_float_t)0.5;
    }

    /* Compute absolute deviations from median */
    ht_float_t dev[HT_HAMPEL_WINDOW_SIZE];
    for (uint8_t i = 0; i < n; ++i) {
        dev[i] = HT_FABS(sorted[i] - med);
    }
    for (uint8_t i = 1; i < n; ++i) {
        ht_float_t key = dev[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && dev[j] > key) {
            dev[j + 1] = dev[j];
            j--;
        }
        dev[j + 1] = key;
    }

    ht_float_t med_dev;
    if ((n % 2U) == 1U) {
        med_dev = dev[n / 2U];
    } else {
        med_dev = (dev[n / 2U - 1U] + dev[n / 2U]) * (ht_float_t)0.5;
    }

    ht_float_t mad = (ht_float_t)1.4826 * med_dev;
    ht_float_t floor_mad = (mad > (ht_float_t)1e-6) ? mad : (ht_float_t)1e-6;
    ht_float_t threshold = f->n_sigma * floor_mad;

    if (HT_FABS(val - med) > threshold) {
        /* Outlier spike detected! Replace in circular window so it does not pollute next samples */
        f->window[feat_idx][idx] = med;
        f->outliers_replaced[feat_idx]++;
        return med;
    }

    return val;
}

/* ========================================================================= */
/* --- 3. Space-Saving Categorical Encoder Implementation ------------------ */
/* ========================================================================= */

void ht_encoder_init(ht_encoder_freq_t *enc) {
    if (enc == NULL) return;
    memset(enc, 0, sizeof(*enc));
    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        enc->feat[f].top_k = (uint8_t)HT_MAX_CATEGORIES;
        enc->feat[f].mode = HT_ENC_NONE;
    }
}

void ht_encoder_set_mode(ht_encoder_freq_t *enc, uint8_t feat_idx, ht_encoder_mode_t mode, uint8_t top_k) {
    if (enc == NULL || feat_idx >= HT_MAX_FEATURES) return;
    enc->feat[feat_idx].mode = mode;
    enc->feat[feat_idx].top_k = (top_k > 0 && top_k <= HT_PREP_MAX_CAT_SLOTS) ? top_k : (uint8_t)HT_MAX_CATEGORIES;
}

void ht_encoder_update_one(ht_encoder_freq_t *enc, uint8_t feat_idx, int32_t cat_val, ht_float_t weight) {
    if (enc == NULL || feat_idx >= HT_MAX_FEATURES || cat_val < 0 || weight <= (ht_float_t)0.0) {
        return;
    }
    ht_feature_encoder_t *fe = &enc->feat[feat_idx];
    if (fe->mode == HT_ENC_NONE) return;

    fe->total_weight += weight;

    /* 1. Check if category already has an assigned slot */
    for (uint8_t i = 0; i < HT_PREP_MAX_CAT_SLOTS; ++i) {
        if (fe->slots[i].occupied && fe->slots[i].key == cat_val) {
            fe->slots[i].count += weight;
            return;
        }
    }

    /* 2. Check for an empty slot */
    for (uint8_t i = 0; i < HT_PREP_MAX_CAT_SLOTS; ++i) {
        if (!fe->slots[i].occupied) {
            fe->slots[i].key = cat_val;
            fe->slots[i].count = weight;
            fe->slots[i].occupied = 1U;
            return;
        }
    }

    /* 3. Space-Saving eviction of min count */
    uint8_t min_idx = 0;
    ht_float_t min_count = fe->slots[0].count;
    for (uint8_t i = 1; i < HT_PREP_MAX_CAT_SLOTS; ++i) {
        if (fe->slots[i].count < min_count) {
            min_count = fe->slots[i].count;
            min_idx = i;
        }
    }
    fe->slots[min_idx].key = cat_val;
    fe->slots[min_idx].count += weight;
}

ht_value_t ht_encoder_transform_one(const ht_encoder_freq_t *enc, uint8_t feat_idx, int32_t cat_val) {
    ht_value_t out;
    out.cat = cat_val;
    if (enc == NULL || feat_idx >= HT_MAX_FEATURES || cat_val < 0) {
        return out;
    }

    const ht_feature_encoder_t *fe = &enc->feat[feat_idx];
    if (fe->mode == HT_ENC_FREQ_RATIO) {
        ht_float_t count = (ht_float_t)0.0;
        for (uint8_t i = 0; i < HT_PREP_MAX_CAT_SLOTS; ++i) {
            if (fe->slots[i].occupied && fe->slots[i].key == cat_val) {
                count = fe->slots[i].count;
                break;
            }
        }
        out.num = (fe->total_weight > (ht_float_t)0.0) ? (count / fe->total_weight) : (ht_float_t)0.0;
        return out;
    } else if (fe->mode == HT_ENC_TOP_K_CAT) {
        uint8_t K = fe->top_k;
        if (K == 0 || K > HT_PREP_MAX_CAT_SLOTS) {
            K = (uint8_t)HT_MAX_CATEGORIES;
        }

        ht_cat_slot_t temp[HT_PREP_MAX_CAT_SLOTS];
        uint8_t n_occ = 0;
        for (uint8_t i = 0; i < HT_PREP_MAX_CAT_SLOTS; ++i) {
            if (fe->slots[i].occupied) {
                temp[n_occ++] = fe->slots[i];
            }
        }

        /* Sort descending by count; break ties with smaller key value */
        for (uint8_t i = 1; i < n_occ; ++i) {
            ht_cat_slot_t key = temp[i];
            int32_t j = (int32_t)i - 1;
            while (j >= 0 && (temp[j].count < key.count || (temp[j].count == key.count && temp[j].key > key.key))) {
                temp[j + 1] = temp[j];
                j--;
            }
            temp[j + 1] = key;
        }

        /* Check if cat_val is among top K-1 categories */
        for (uint8_t r = 0; r < (uint8_t)(K - 1U) && r < n_occ; ++r) {
            if (temp[r].key == cat_val) {
                out.cat = (int32_t)r;
                return out;
            }
        }

        /* All other categories map to the K-1 bucket ("Others") */
        out.cat = (int32_t)(K - 1U);
        return out;
    }

    return out;
}

ht_value_t ht_encoder_learn_transform_one(ht_encoder_freq_t *enc, uint8_t feat_idx, int32_t cat_val, ht_float_t weight) {
    ht_encoder_update_one(enc, feat_idx, cat_val, weight);
    return ht_encoder_transform_one(enc, feat_idx, cat_val);
}

/* ========================================================================= */
/* --- 4. Ben-Haim & Tom-Tov Quantizer / Discretizer Implementation -------- */
/* ========================================================================= */

void ht_quantizer_init(ht_quantizer_t *q) {
    if (q == NULL) return;
    memset(q, 0, sizeof(*q));
    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        q->feat[f].min_val = HT_INFINITY;
        q->feat[f].max_val = -HT_INFINITY;
        q->feat[f].n_bins = 5U;
        q->feat[f].mode = HT_QUANT_NONE;
    }
}

void ht_quantizer_set_mode(ht_quantizer_t *q, uint8_t feat_idx, ht_quant_mode_t mode, uint8_t n_bins) {
    if (q == NULL || feat_idx >= HT_MAX_FEATURES) return;
    q->feat[feat_idx].mode = mode;
    q->feat[feat_idx].n_bins = (n_bins > 0) ? n_bins : 5U;
}

void ht_quantizer_update_one(ht_quantizer_t *q, uint8_t feat_idx, ht_float_t val, ht_float_t weight) {
    if (q == NULL || feat_idx >= HT_MAX_FEATURES || HT_ISNAN(val) || weight <= (ht_float_t)0.0) {
        return;
    }
    ht_feature_quantizer_t *fq = &q->feat[feat_idx];
    if (fq->mode == HT_QUANT_NONE) return;

    fq->total_weight += weight;
    if (val < fq->min_val) fq->min_val = val;
    if (val > fq->max_val) fq->max_val = val;

    /* Check if an exact centroid exists */
    for (uint8_t i = 0; i < fq->n_centroids; ++i) {
        if (HT_FABS(fq->centroids[i].mean - val) <= (ht_float_t)1e-9) {
            fq->centroids[i].weight += weight;
            return;
        }
    }

    if (fq->n_centroids < HT_PREP_MAX_CENTROIDS) {
        /* Insert in sorted order */
        uint8_t ins_idx = fq->n_centroids;
        for (uint8_t i = 0; i < fq->n_centroids; ++i) {
            if (val < fq->centroids[i].mean) {
                ins_idx = i;
                break;
            }
        }
        for (int32_t j = (int32_t)fq->n_centroids - 1; j >= (int32_t)ins_idx; --j) {
            fq->centroids[j + 1] = fq->centroids[j];
        }
        fq->centroids[ins_idx].mean = val;
        fq->centroids[ins_idx].weight = weight;
        fq->n_centroids++;
    } else {
        /* Temporary buffer of HT_PREP_MAX_CENTROIDS + 1 elements */
        ht_centroid_t temp[HT_PREP_MAX_CENTROIDS + 1U];
        uint8_t ins_idx = HT_PREP_MAX_CENTROIDS;
        for (uint8_t i = 0; i < HT_PREP_MAX_CENTROIDS; ++i) {
            if (val < fq->centroids[i].mean) {
                ins_idx = i;
                break;
            }
        }
        for (uint8_t i = 0; i < ins_idx; ++i) {
            temp[i] = fq->centroids[i];
        }
        temp[ins_idx].mean = val;
        temp[ins_idx].weight = weight;
        for (uint8_t i = ins_idx; i < HT_PREP_MAX_CENTROIDS; ++i) {
            temp[i + 1U] = fq->centroids[i];
        }

        /* Find adjacent pair with minimum distance */
        uint8_t min_idx = 0;
        ht_float_t min_dist = temp[1].mean - temp[0].mean;
        for (uint8_t i = 1; i < HT_PREP_MAX_CENTROIDS; ++i) {
            ht_float_t d = temp[i + 1U].mean - temp[i].mean;
            if (d < min_dist) {
                min_dist = d;
                min_idx = i;
            }
        }

        /* Merge pair min_idx and min_idx + 1 */
        ht_float_t m_comb = temp[min_idx].weight + temp[min_idx + 1U].weight;
        ht_float_t p_comb = (temp[min_idx].mean * temp[min_idx].weight +
                             temp[min_idx + 1U].mean * temp[min_idx + 1U].weight) / m_comb;

        /* Write back HT_PREP_MAX_CENTROIDS elements */
        for (uint8_t i = 0; i < min_idx; ++i) {
            fq->centroids[i] = temp[i];
        }
        fq->centroids[min_idx].mean = p_comb;
        fq->centroids[min_idx].weight = m_comb;
        for (uint8_t i = min_idx + 1U; i < HT_PREP_MAX_CENTROIDS; ++i) {
            fq->centroids[i] = temp[i + 1U];
        }
    }
}

ht_float_t ht_quantizer_cdf(const ht_feature_quantizer_t *fq, ht_float_t val) {
    if (fq == NULL || fq->n_centroids == 0 || fq->total_weight <= (ht_float_t)0.0 || HT_ISNAN(val)) {
        return (ht_float_t)0.5;
    }

    if (fq->n_centroids == 1U) {
        if (val < fq->centroids[0].mean) return (ht_float_t)0.0;
        if (val > fq->centroids[0].mean) return (ht_float_t)1.0;
        return (ht_float_t)0.5;
    }

    if (val <= fq->centroids[0].mean) {
        return (ht_float_t)0.0;
    }
    if (val >= fq->centroids[fq->n_centroids - 1U].mean) {
        return (ht_float_t)1.0;
    }

    /* Locate segment p_i <= val < p_{i+1} */
    uint8_t i = 0;
    for (uint8_t k = 0; k < fq->n_centroids - 1U; ++k) {
        if (val >= fq->centroids[k].mean && val < fq->centroids[k + 1U].mean) {
            i = k;
            break;
        }
    }

    ht_float_t w_prev = (ht_float_t)0.0;
    for (uint8_t j = 0; j < i; ++j) {
        w_prev += fq->centroids[j].weight;
    }

    ht_float_t p_i = fq->centroids[i].mean;
    ht_float_t p_next = fq->centroids[i + 1U].mean;
    ht_float_t m_i = fq->centroids[i].weight;
    ht_float_t m_next = fq->centroids[i + 1U].weight;
    ht_float_t dp = p_next - p_i;

    ht_float_t alpha = (dp > (ht_float_t)1e-12) ? ((val - p_i) / dp) : (ht_float_t)0.0;
    ht_float_t mass = w_prev + (m_i * (ht_float_t)0.5) + alpha * ((m_i + m_next) * (ht_float_t)0.5);

    ht_float_t cdf = mass / fq->total_weight;
    if (cdf < (ht_float_t)0.0) cdf = (ht_float_t)0.0;
    if (cdf > (ht_float_t)1.0) cdf = (ht_float_t)1.0;
    return cdf;
}

ht_float_t ht_quantizer_quantile(const ht_feature_quantizer_t *fq, ht_float_t q_val) {
    if (fq == NULL || fq->n_centroids == 0) {
        return (ht_float_t)0.0;
    }
    if (q_val <= (ht_float_t)0.0) {
        return fq->centroids[0].mean;
    }
    if (q_val >= (ht_float_t)1.0) {
        return fq->centroids[fq->n_centroids - 1U].mean;
    }

    ht_float_t target_mass = q_val * fq->total_weight;
    ht_float_t cum = (ht_float_t)0.0;

    for (uint8_t i = 0; i < fq->n_centroids - 1U; ++i) {
        ht_float_t m_i = fq->centroids[i].weight;
        ht_float_t m_next = fq->centroids[i + 1U].weight;
        ht_float_t seg_mass = (m_i + m_next) * (ht_float_t)0.5;

        if (cum + (m_i * (ht_float_t)0.5) + seg_mass >= target_mass) {
            ht_float_t needed = target_mass - cum - (m_i * (ht_float_t)0.5);
            if (needed < (ht_float_t)0.0) needed = (ht_float_t)0.0;
            ht_float_t alpha = (seg_mass > (ht_float_t)1e-12) ? (needed / seg_mass) : (ht_float_t)0.0;
            if (alpha > (ht_float_t)1.0) alpha = (ht_float_t)1.0;
            return fq->centroids[i].mean + alpha * (fq->centroids[i + 1U].mean - fq->centroids[i].mean);
        }
        cum += m_i;
    }

    return fq->centroids[fq->n_centroids - 1U].mean;
}

ht_value_t ht_quantizer_transform_one(const ht_quantizer_t *q, uint8_t feat_idx, ht_float_t val) {
    ht_value_t out;
    out.num = val;
    if (q == NULL || feat_idx >= HT_MAX_FEATURES || HT_ISNAN(val)) {
        return out;
    }

    const ht_feature_quantizer_t *fq = &q->feat[feat_idx];
    if (fq->mode == HT_QUANT_CDF_UNIFORM) {
        out.num = ht_quantizer_cdf(fq, val);
        return out;
    } else if (fq->mode == HT_QUANT_KBINS_ORDINAL) {
        ht_float_t cdf = ht_quantizer_cdf(fq, val);
        uint8_t n_bins = fq->n_bins;
        int32_t b = (int32_t)(cdf * (ht_float_t)n_bins);
        if (b >= (int32_t)n_bins) b = (int32_t)n_bins - 1;
        if (b < 0) b = 0;
        out.num = (ht_float_t)b;
        return out;
    } else if (fq->mode == HT_QUANT_KBINS_CAT) {
        ht_float_t cdf = ht_quantizer_cdf(fq, val);
        uint8_t n_bins = fq->n_bins;
        int32_t b = (int32_t)(cdf * (ht_float_t)n_bins);
        if (b >= (int32_t)n_bins) b = (int32_t)n_bins - 1;
        if (b < 0) b = 0;
        out.cat = b;
        return out;
    }

    return out;
}

ht_value_t ht_quantizer_learn_transform_one(ht_quantizer_t *q, uint8_t feat_idx, ht_float_t val, ht_float_t weight) {
    ht_quantizer_update_one(q, feat_idx, val, weight);
    return ht_quantizer_transform_one(q, feat_idx, val);
}

/* ========================================================================= */
/* --- 5. Composable Micro-Pipeline Implementation ------------------------- */
/* ========================================================================= */

void ht_prep_pipeline_init(ht_prep_pipeline_t *pipe, uint8_t n_features, const ht_feature_type_t feature_types[HT_MAX_FEATURES]) {
    if (pipe == NULL) return;
    memset(pipe, 0, sizeof(*pipe));
    pipe->n_features = (n_features > HT_MAX_FEATURES) ? (uint8_t)HT_MAX_FEATURES : n_features;

    if (feature_types != NULL) {
        for (uint8_t f = 0; f < pipe->n_features; ++f) {
            pipe->feature_types_in[f] = feature_types[f];
            pipe->feature_types_out[f] = feature_types[f];
        }
    }

    ht_hampel_init(&pipe->hampel);
    ht_encoder_init(&pipe->encoder);
    ht_quantizer_init(&pipe->quantizer);
    ht_scaler_init(&pipe->scaler);
}

void ht_prep_pipeline_update_types_out(ht_prep_pipeline_t *pipe) {
    if (pipe == NULL) return;
    for (uint8_t f = 0; f < pipe->n_features; ++f) {
        ht_feature_type_t t = pipe->feature_types_in[f];
        if (t == HT_FEAT_CATEGORICAL) {
            if (pipe->encoder.feat[f].mode == HT_ENC_FREQ_RATIO) {
                t = HT_FEAT_NUMERIC;
            }
        } else {
            if (pipe->quantizer.feat[f].mode == HT_QUANT_KBINS_CAT) {
                t = HT_FEAT_CATEGORICAL;
            }
        }
        pipe->feature_types_out[f] = t;
    }
}

void ht_prep_transform_one(const ht_prep_pipeline_t *pipe, const ht_value_t x_in[HT_MAX_FEATURES], ht_value_t x_out[HT_MAX_FEATURES]) {
    if (pipe == NULL || x_in == NULL || x_out == NULL) return;

    for (uint8_t f = 0; f < pipe->n_features; ++f) {
        ht_value_t cur = x_in[f];

        if (pipe->feature_types_in[f] == HT_FEAT_NUMERIC) {
            /* 1. Hampel filter */
            if (pipe->hampel.enabled[f] && !HT_ISNAN(cur.num)) {
                /* Transform mode: inspect without adding to history (or evaluate against current window) */
                /* For pure deterministic transform, evaluate against current window state */
                uint8_t n = pipe->hampel.count[f];
                if (n >= 3U) {
                    ht_float_t sorted[HT_HAMPEL_WINDOW_SIZE];
                    for (uint8_t i = 0; i < n; ++i) sorted[i] = pipe->hampel.window[f][i];
                    for (uint8_t i = 1; i < n; ++i) {
                        ht_float_t key = sorted[i]; int32_t j = (int32_t)i - 1;
                        while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j]; j--; }
                        sorted[j + 1] = key;
                    }
                    ht_float_t med = ((n % 2U) == 1U) ? sorted[n / 2U] : (sorted[n / 2U - 1U] + sorted[n / 2U]) * (ht_float_t)0.5;
                    ht_float_t dev[HT_HAMPEL_WINDOW_SIZE];
                    for (uint8_t i = 0; i < n; ++i) dev[i] = HT_FABS(sorted[i] - med);
                    for (uint8_t i = 1; i < n; ++i) {
                        ht_float_t key = dev[i]; int32_t j = (int32_t)i - 1;
                        while (j >= 0 && dev[j] > key) { dev[j + 1] = dev[j]; j--; }
                        dev[j + 1] = key;
                    }
                    ht_float_t med_dev = ((n % 2U) == 1U) ? dev[n / 2U] : (dev[n / 2U - 1U] + dev[n / 2U]) * (ht_float_t)0.5;
                    ht_float_t mad = (ht_float_t)1.4826 * med_dev;
                    ht_float_t floor_mad = (mad > (ht_float_t)1e-6) ? mad : (ht_float_t)1e-6;
                    if (HT_FABS(cur.num - med) > pipe->hampel.n_sigma * floor_mad) {
                        cur.num = med;
                    }
                }
            }

            /* 2. Quantizer */
            if (pipe->quantizer.feat[f].mode != HT_QUANT_NONE && !HT_ISNAN(cur.num)) {
                cur = ht_quantizer_transform_one(&pipe->quantizer, f, cur.num);
            }

            /* 3. Scaler (if feature is still numeric) */
            if (pipe->feature_types_out[f] == HT_FEAT_NUMERIC && pipe->scaler.mode[f] != HT_SCALER_NONE && !HT_ISNAN(cur.num)) {
                cur.num = ht_scaler_transform_one(&pipe->scaler, f, cur.num);
            }
        } else {
            /* 1. Categorical Space-Saving Encoder */
            if (pipe->encoder.feat[f].mode != HT_ENC_NONE) {
                cur = ht_encoder_transform_one(&pipe->encoder, f, cur.cat);
            }

            /* 2. Scaler (if converted to continuous numeric frequency ratio) */
            if (pipe->feature_types_out[f] == HT_FEAT_NUMERIC && pipe->scaler.mode[f] != HT_SCALER_NONE && !HT_ISNAN(cur.num)) {
                cur.num = ht_scaler_transform_one(&pipe->scaler, f, cur.num);
            }
        }

        x_out[f] = cur;
    }
}

void ht_prep_learn_transform_one(ht_prep_pipeline_t *pipe, const ht_value_t x_in[HT_MAX_FEATURES], ht_value_t x_out[HT_MAX_FEATURES], ht_float_t weight) {
    if (pipe == NULL || x_in == NULL || x_out == NULL || weight <= (ht_float_t)0.0) {
        return;
    }

    for (uint8_t f = 0; f < pipe->n_features; ++f) {
        ht_value_t cur = x_in[f];

        if (pipe->feature_types_in[f] == HT_FEAT_NUMERIC) {
            /* 1. Hampel filter learn-transform */
            if (pipe->hampel.enabled[f] && !HT_ISNAN(cur.num)) {
                cur.num = ht_hampel_filter_one(&pipe->hampel, f, cur.num);
            }

            /* 2. Quantizer update + transform */
            if (pipe->quantizer.feat[f].mode != HT_QUANT_NONE && !HT_ISNAN(cur.num)) {
                cur = ht_quantizer_learn_transform_one(&pipe->quantizer, f, cur.num, weight);
            }

            /* 3. Scaler (if feature remains numeric) */
            if (pipe->feature_types_out[f] == HT_FEAT_NUMERIC && pipe->scaler.mode[f] != HT_SCALER_NONE && !HT_ISNAN(cur.num)) {
                cur.num = ht_scaler_learn_transform_one(&pipe->scaler, f, cur.num, weight);
            }
        } else {
            /* 1. Categorical encoder learn-transform */
            if (pipe->encoder.feat[f].mode != HT_ENC_NONE) {
                cur = ht_encoder_learn_transform_one(&pipe->encoder, f, cur.cat, weight);
            }

            /* 2. Scaler (if converted to continuous numeric) */
            if (pipe->feature_types_out[f] == HT_FEAT_NUMERIC && pipe->scaler.mode[f] != HT_SCALER_NONE && !HT_ISNAN(cur.num)) {
                cur.num = ht_scaler_learn_transform_one(&pipe->scaler, f, cur.num, weight);
            }
        }

        x_out[f] = cur;
    }
}
