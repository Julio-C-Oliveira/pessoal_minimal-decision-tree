/**
 * @file ht_voting.h
 * @brief Shared voting and prequential error weighting engine for ensembles.
 *
 * Shared across Phase 9 (Local Micro-Ensembles) and Phase 11 (Fed-Forest).
 * Supports uniform soft-voting and inverse-error-squared weighting for both
 * classification and continuous regression.
 */

#ifndef HT_VOTING_H
#define HT_VOTING_H

#include "ht_config.h"

/**
 * @brief Ensemble voting / aggregation strategy.
 */
typedef enum {
    HT_VOTE_WEIGHTED = 0, /**< w_m = 1 / (err_m^2 + eps) for classif, 1 / (fMSE_m + eps) for reg */
    HT_VOTE_UNIFORM  = 1  /**< Equal weights w_m = 1.0 */
} ht_voting_strategy_t;

/**
 * @brief Updates exponential faded classification error for one ensemble member.
 *
 * e_t = (pred_class != true_class) ? 1.0 : 0.0
 * faded_err <- decay * faded_err + (1 - decay) * e_t
 */
void ht_voting_update_error_classif(ht_float_t *faded_err,
                                    uint8_t pred_class,
                                    uint8_t true_class,
                                    ht_float_t decay);

/**
 * @brief Updates exponential faded MSE for one regression ensemble member.
 *
 * fMSE <- decay * fMSE + (1 - decay) * (y_true - y_pred)^2
 */
void ht_voting_update_error_reg(ht_float_t *faded_mse,
                                ht_float_t y_pred,
                                ht_float_t y_true,
                                ht_float_t decay);

/**
 * @brief Computes the voting weight for a member given its faded error.
 */
ht_float_t ht_voting_member_weight(ht_voting_strategy_t strategy,
                                   ht_task_type_t task_type,
                                   ht_float_t faded_error);

/**
 * @brief Combines member class probability vectors using weighted soft-voting.
 *
 * @param n_members Number of active committee members.
 * @param member_probas Flat array of size n_members * HT_MAX_CLASSES.
 * @param member_weights Array of size n_members (non-negative weights).
 * @param n_classes Number of active classes.
 * @param proba_out Output normalized probability vector.
 * @return HT_OK on success, or HT_ERR_INVALID_ARG.
 */
ht_status_t ht_voting_combine_proba(uint16_t n_members,
                                    const ht_float_t *member_probas,
                                    const ht_float_t *member_weights,
                                    uint8_t n_classes,
                                    ht_float_t proba_out[HT_MAX_CLASSES]);

/**
 * @brief Combines member continuous regression predictions via weighted average.
 */
ht_float_t ht_voting_combine_reg(uint16_t n_members,
                                 const ht_float_t *member_preds,
                                 const ht_float_t *member_weights);

#endif /* HT_VOTING_H */
