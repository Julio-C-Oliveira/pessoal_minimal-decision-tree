/**
 * @file ht_voting.c
 * @brief Shared voting and prequential error weighting engine implementation.
 */

#include "ht_voting.h"

#define HT_VOTE_EPS HT_CONST(1e-4)

void ht_voting_update_error_classif(ht_float_t *faded_err,
                                    uint8_t pred_class,
                                    uint8_t true_class,
                                    ht_float_t decay) {
    if (faded_err == NULL) {
        return;
    }
    ht_float_t err = (pred_class != true_class) ? HT_CONST(1.0) : HT_CONST(0.0);
    ht_float_t one_m_decay = HT_SUB(HT_CONST(1.0), decay);
    *faded_err = HT_ADD(HT_MUL(decay, *faded_err), HT_MUL(one_m_decay, err));
}

void ht_voting_update_error_reg(ht_float_t *faded_mse,
                                ht_float_t y_pred,
                                ht_float_t y_true,
                                ht_float_t decay) {
    if (faded_mse == NULL || HT_ISNAN(y_pred) || HT_ISNAN(y_true)) {
        return;
    }
#ifdef HT_USE_FIXED_POINT
    /* Clamp squared error and faded MSE strictly below HT_INFINITY (32000.0 in Q16.16)
     * so high-amplitude targets (Bikes, Appliances, NYC Taxi) never overflow into
     * +HT_INFINITY and zero out member weights in ht_voting_member_weight. */
    int64_t diff = (int64_t)y_true - (int64_t)y_pred;
    uint64_t ad = (diff < 0) ? (uint64_t)(-diff) : (uint64_t)diff;
    uint64_t sq_q16 = (ad * ad + 32768ULL) >> 16;
    const uint64_t max_mse_q16 = (uint64_t)HT_CONST(32000.0);
    if (sq_q16 > max_mse_q16) {
        sq_q16 = max_mse_q16;
    }
    int64_t d_q16 = (int64_t)decay;
    if (d_q16 < 0) d_q16 = 0;
    if (d_q16 > (int64_t)HT_FIXED_ONE) d_q16 = (int64_t)HT_FIXED_ONE;
    uint64_t one_m_d_q16 = (uint64_t)((int64_t)HT_FIXED_ONE - d_q16);
    uint64_t prev = (*faded_mse > 0) ? (uint64_t)(uint32_t)*faded_mse : 0ULL;
    if (prev > max_mse_q16) {
        prev = max_mse_q16;
    }
    uint64_t next = ((prev * (uint64_t)d_q16 + 32768ULL) >> 16) +
                    ((sq_q16 * one_m_d_q16 + 32768ULL) >> 16);
    if (next > max_mse_q16) {
        next = max_mse_q16;
    }
    *faded_mse = (ht_float_t)next;
#else
    ht_float_t diff = HT_SUB(y_true, y_pred);
    ht_float_t sq = HT_MUL(diff, diff);
    ht_float_t one_m_decay = HT_SUB(HT_CONST(1.0), decay);
    *faded_mse = HT_ADD(HT_MUL(decay, *faded_mse), HT_MUL(one_m_decay, sq));
#endif
}

ht_float_t ht_voting_member_weight(ht_voting_strategy_t strategy,
                                   ht_task_type_t task_type,
                                   ht_float_t faded_error) {
    if (strategy == HT_VOTE_UNIFORM) {
        return HT_CONST(1.0);
    }
#ifdef HT_USE_FIXED_POINT
    uint64_t e = (faded_error > 0) ? (uint64_t)(uint32_t)faded_error : 0ULL;
    if (e > (uint64_t)HT_CONST(32000.0)) {
        e = (uint64_t)HT_CONST(32000.0);
    }
    if (task_type == HT_TASK_REGRESSION) {
        /* 1 / (MSE + eps) in Q16.16 with strict positive floor >= 1 ULP so valid
         * high-amplitude regression models never yield zero total weight. */
        uint64_t denom = e + (uint64_t)HT_VOTE_EPS;
        if (denom < 1ULL) denom = 1ULL;
        uint64_t w = ((uint64_t)HT_FIXED_ONE << 16) / denom;
        if (w < 1ULL) w = 1ULL;
        if (w > (uint64_t)HT_CONST(16384.0)) w = (uint64_t)HT_CONST(16384.0);
        return (ht_float_t)w;
    }
    uint64_t e2_q16 = (e * e + 32768ULL) >> 16;
    uint64_t denom = e2_q16 + (uint64_t)HT_VOTE_EPS;
    if (denom < 1ULL) denom = 1ULL;
    uint64_t w = ((uint64_t)HT_FIXED_ONE << 16) / denom;
    if (w < 1ULL) w = 1ULL;
    if (w > (uint64_t)HT_CONST(16384.0)) w = (uint64_t)HT_CONST(16384.0);
    return (ht_float_t)w;
#else
    ht_float_t e = (faded_error > HT_CONST(0.0)) ? faded_error : HT_CONST(0.0);
    if (task_type == HT_TASK_REGRESSION) {
        return HT_DIV(HT_CONST(1.0), HT_ADD(e, HT_VOTE_EPS));
    }
    return HT_DIV(HT_CONST(1.0), HT_ADD(HT_MUL(e, e), HT_VOTE_EPS));
#endif
}

ht_status_t ht_voting_combine_proba(uint16_t n_members,
                                    const ht_float_t *member_probas,
                                    const ht_float_t *member_weights,
                                    uint8_t n_classes,
                                    ht_float_t proba_out[HT_MAX_CLASSES]) {
    if (n_members == 0U || member_probas == NULL || member_weights == NULL ||
        proba_out == NULL || n_classes == 0U || n_classes > HT_MAX_CLASSES) {
        return HT_ERR_INVALID_ARG;
    }

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        proba_out[c] = HT_CONST(0.0);
    }

#ifdef HT_USE_FIXED_POINT
    int64_t w_sum = 0;
    int64_t proba_acc[HT_MAX_CLASSES];
    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        proba_acc[c] = 0;
    }

    for (uint16_t m = 0; m < n_members; ++m) {
        ht_float_t w = member_weights[m];
        if (w <= HT_CONST(0.0) || HT_ISNAN(w)) {
            continue;
        }
        w_sum += (int64_t)w;
        const ht_float_t *pm = &member_probas[(size_t)m * HT_MAX_CLASSES];
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_acc[c] += (int64_t)w * (int64_t)pm[c];
        }
    }

    if (w_sum <= 0) {
        ht_float_t uni = HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_classes));
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_out[c] = uni;
        }
        return HT_OK;
    }

    for (uint8_t c = 0; c < n_classes; ++c) {
        proba_out[c] = ht_fixed_saturate64(proba_acc[c] / w_sum);
    }
    return HT_OK;
#else
    ht_float_t w_sum = HT_CONST(0.0);
    for (uint16_t m = 0; m < n_members; ++m) {
        ht_float_t w = member_weights[m];
        if (w <= HT_CONST(0.0) || HT_ISNAN(w)) {
            continue;
        }
        w_sum = HT_ADD(w_sum, w);
        const ht_float_t *pm = &member_probas[(size_t)m * HT_MAX_CLASSES];
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_out[c] = HT_ADD(proba_out[c], HT_MUL(w, pm[c]));
        }
    }

    if (w_sum <= HT_CONST(0.0)) {
        ht_float_t uni = HT_DIV(HT_CONST(1.0), HT_FROM_UINT(n_classes));
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_out[c] = uni;
        }
        return HT_OK;
    }

    for (uint8_t c = 0; c < n_classes; ++c) {
        proba_out[c] = HT_DIV(proba_out[c], w_sum);
    }
    return HT_OK;
#endif
}

ht_float_t ht_voting_combine_reg(uint16_t n_members,
                                 const ht_float_t *member_preds,
                                 const ht_float_t *member_weights) {
    if (n_members == 0U || member_preds == NULL || member_weights == NULL) {
        return HT_CONST(0.0);
    }

#ifdef HT_USE_FIXED_POINT
    int64_t num = 0;
    int64_t den = 0;
    for (uint16_t m = 0; m < n_members; ++m) {
        ht_float_t w = member_weights[m];
        if (w <= HT_CONST(0.0) || HT_ISNAN(w) || HT_ISNAN(member_preds[m])) {
            continue;
        }
        num += (int64_t)w * (int64_t)member_preds[m];
        den += (int64_t)w;
    }

    if (den <= 0) {
        return HT_CONST(0.0);
    }
    return ht_fixed_saturate64(num / den);
#else
    ht_float_t num = HT_CONST(0.0);
    ht_float_t den = HT_CONST(0.0);
    for (uint16_t m = 0; m < n_members; ++m) {
        ht_float_t w = member_weights[m];
        if (w <= HT_CONST(0.0) || HT_ISNAN(w) || HT_ISNAN(member_preds[m])) {
            continue;
        }
        num = HT_ADD(num, HT_MUL(w, member_preds[m]));
        den = HT_ADD(den, w);
    }

    if (den <= HT_CONST(0.0)) {
        return HT_CONST(0.0);
    }
    return HT_DIV(num, den);
#endif
}
