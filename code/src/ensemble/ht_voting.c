/**
 * @file ht_voting.c
 * @brief Shared voting and prequential error weighting engine implementation.
 */

#include "ht_voting.h"

#define HT_VOTE_EPS ((ht_float_t)1e-4)

void ht_voting_update_error_classif(ht_float_t *faded_err,
                                    uint8_t pred_class,
                                    uint8_t true_class,
                                    ht_float_t decay) {
    if (faded_err == NULL) {
        return;
    }
    ht_float_t err = (pred_class != true_class) ? (ht_float_t)1.0 : (ht_float_t)0.0;
    *faded_err = decay * (*faded_err) + ((ht_float_t)1.0 - decay) * err;
}

void ht_voting_update_error_reg(ht_float_t *faded_mse,
                                ht_float_t y_pred,
                                ht_float_t y_true,
                                ht_float_t decay) {
    if (faded_mse == NULL || HT_ISNAN(y_pred) || HT_ISNAN(y_true)) {
        return;
    }
    ht_float_t diff = y_true - y_pred;
    ht_float_t sq = diff * diff;
    *faded_mse = decay * (*faded_mse) + ((ht_float_t)1.0 - decay) * sq;
}

ht_float_t ht_voting_member_weight(ht_voting_strategy_t strategy,
                                   ht_task_type_t task_type,
                                   ht_float_t faded_error) {
    if (strategy == HT_VOTE_UNIFORM) {
        return (ht_float_t)1.0;
    }
    ht_float_t e = (faded_error > (ht_float_t)0.0) ? faded_error : (ht_float_t)0.0;
    if (task_type == HT_TASK_REGRESSION) {
        return (ht_float_t)1.0 / (e + HT_VOTE_EPS);
    }
    return (ht_float_t)1.0 / ((e * e) + HT_VOTE_EPS);
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
        proba_out[c] = (ht_float_t)0.0;
    }

    ht_float_t w_sum = (ht_float_t)0.0;
    for (uint16_t m = 0; m < n_members; ++m) {
        ht_float_t w = member_weights[m];
        if (w <= (ht_float_t)0.0 || HT_ISNAN(w)) {
            continue;
        }
        w_sum += w;
        const ht_float_t *pm = &member_probas[(size_t)m * HT_MAX_CLASSES];
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_out[c] += w * pm[c];
        }
    }

    if (w_sum <= (ht_float_t)0.0) {
        ht_float_t uni = (ht_float_t)1.0 / (ht_float_t)n_classes;
        for (uint8_t c = 0; c < n_classes; ++c) {
            proba_out[c] = uni;
        }
        return HT_OK;
    }

    for (uint8_t c = 0; c < n_classes; ++c) {
        proba_out[c] /= w_sum;
    }
    return HT_OK;
}

ht_float_t ht_voting_combine_reg(uint16_t n_members,
                                 const ht_float_t *member_preds,
                                 const ht_float_t *member_weights) {
    if (n_members == 0U || member_preds == NULL || member_weights == NULL) {
        return (ht_float_t)0.0;
    }

    ht_float_t num = (ht_float_t)0.0;
    ht_float_t den = (ht_float_t)0.0;
    for (uint16_t m = 0; m < n_members; ++m) {
        ht_float_t w = member_weights[m];
        if (w <= (ht_float_t)0.0 || HT_ISNAN(w) || HT_ISNAN(member_preds[m])) {
            continue;
        }
        num += w * member_preds[m];
        den += w;
    }

    if (den <= (ht_float_t)0.0) {
        return (ht_float_t)0.0;
    }
    return num / den;
}
