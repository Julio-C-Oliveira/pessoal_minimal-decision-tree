/**
 * @file ht_fed_forest.h
 * @brief Mode 1 Federated Learning: Federated Forest (Fed-Forest) in pure C11.
 *
 * Each embedded client trains 1 full local tree (VFDT/HAT/VFDT-R/HAT-R) and
 * exports a BFS-compacted inference-only tree (~1.3 KB for 64 nodes) to the
 * static zero-malloc server. The server selects up to M peer trees using a
 * configurable policy (ALL, TOP_M, ROUND_ROBIN) and returns a compact forest
 * bundle. Clients perform test-then-train weighted voting over local + peer trees.
 */

#ifndef HT_FED_FOREST_H
#define HT_FED_FOREST_H

#include "ht_config.h"
#include "ht_tree.h"
#include "ht_compact_tree.h"
#include "ht_voting.h"
#include "ht_fed_proto.h"

/**
 * @brief Server tree selection policy when K > M.
 */
typedef enum {
    HT_FED_FOREST_AGG_ALL         = 0, /**< Return all stored trees up to M */
    HT_FED_FOREST_AGG_TOP_M       = 1, /**< Return top-M trees with lowest reported prequential error */
    HT_FED_FOREST_AGG_ROUND_ROBIN = 2  /**< Rotate selection window across rounds for Non-IID diversity */
} ht_fed_forest_agg_t;

/**
 * @brief Client configuration for Fed-Forest.
 */
typedef struct {
    ht_config_params_t   tree_params; /**< Local trainable tree parameters */
    ht_voting_strategy_t voting;      /**< HT_VOTE_WEIGHTED (default) or HT_VOTE_UNIFORM */
    ht_float_t           error_decay; /**< Exponential decay alpha for member prequential error (default: 0.98) */
    uint16_t             client_id;   /**< Unique client identifier */
} ht_fed_forest_client_cfg_t;

/**
 * @brief Embedded Fed-Forest client instance (1 trainable tree + M compact peer trees).
 */
typedef struct {
    ht_tree_t            local_tree;
    ht_compact_tree_t    peer_trees[HT_FED_FOREST_MAX_TREES];
    ht_float_t           member_faded_error[1U + HT_FED_FOREST_MAX_TREES];
    ht_float_t           error_decay;
    ht_voting_strategy_t voting;
    uint32_t             round_id;
    uint16_t             client_id;
    uint16_t             peer_client_ids[HT_FED_FOREST_MAX_TREES];
    uint8_t              n_peers;
} ht_fed_forest_client_t;

/**
 * @brief Server configuration for Fed-Forest.
 */
typedef struct {
    ht_task_type_t      task_type;
    ht_fed_forest_agg_t agg_policy;
    uint8_t             max_forest_size; /**< M <= HT_FED_FOREST_MAX_TREES */
} ht_fed_forest_server_cfg_t;

/**
 * @brief One stored client tree entry on the Fed-Forest server.
 */
typedef struct {
    ht_compact_tree_t tree;
    ht_float_t        reported_error;
    ht_float_t        train_weight;
    uint32_t          last_round;
    uint16_t          client_id;
    uint8_t           occupied;
} ht_fed_forest_entry_t;

/**
 * @brief Static zero-malloc Fed-Forest aggregating server.
 */
typedef struct {
    ht_fed_forest_entry_t      entries[HT_FED_SERVER_MAX_TREES];
    ht_fed_forest_server_cfg_t cfg;
    uint32_t                   round_id;
    uint16_t                   n_entries;
    uint16_t                   rr_cursor;
} ht_fed_forest_server_t;

/* --- Client API ---------------------------------------------------------- */

void ht_fed_forest_default_client_config(ht_fed_forest_client_cfg_t *cfg,
                                         uint16_t client_id,
                                         const ht_config_params_t *tree_params);

ht_status_t ht_fed_forest_client_init(ht_fed_forest_client_t *client,
                                      const ht_fed_forest_client_cfg_t *cfg);

ht_status_t ht_fed_forest_client_set_adwin_pool(ht_fed_forest_client_t *client,
                                                ht_adwin_pool_t *pool);

ht_status_t ht_fed_forest_client_predict_proba(const ht_fed_forest_client_t *client,
                                               const ht_value_t x[HT_MAX_FEATURES],
                                               ht_float_t proba_out[HT_MAX_CLASSES]);

uint8_t ht_fed_forest_client_predict(const ht_fed_forest_client_t *client,
                                     const ht_value_t x[HT_MAX_FEATURES]);

ht_status_t ht_fed_forest_client_learn_one(ht_fed_forest_client_t *client,
                                           const ht_value_t x[HT_MAX_FEATURES],
                                           uint8_t y,
                                           ht_float_t weight);

ht_float_t ht_fed_forest_client_predict_reg(const ht_fed_forest_client_t *client,
                                            const ht_value_t x[HT_MAX_FEATURES]);

ht_status_t ht_fed_forest_client_learn_one_reg(ht_fed_forest_client_t *client,
                                               const ht_value_t x[HT_MAX_FEATURES],
                                               ht_float_t y,
                                               ht_float_t weight);

ht_status_t ht_fed_forest_client_export_update(const ht_fed_forest_client_t *client,
                                               void *out_buf,
                                               size_t buf_cap,
                                               size_t *out_written);

ht_status_t ht_fed_forest_client_import_forest(ht_fed_forest_client_t *client,
                                               const void *buf,
                                               size_t buf_len);

/* --- Server API ---------------------------------------------------------- */

void ht_fed_forest_default_server_config(ht_fed_forest_server_cfg_t *cfg,
                                         ht_task_type_t task_type);

ht_status_t ht_fed_forest_server_init(ht_fed_forest_server_t *server,
                                      const ht_fed_forest_server_cfg_t *cfg);

ht_status_t ht_fed_forest_server_ingest(ht_fed_forest_server_t *server,
                                        const void *buf,
                                        size_t buf_len);

ht_status_t ht_fed_forest_server_build_forest(ht_fed_forest_server_t *server,
                                              uint16_t target_client_id,
                                              bool exclude_self,
                                              void *out_buf,
                                              size_t buf_cap,
                                              size_t *out_written);

void ht_fed_forest_server_advance_round(ht_fed_forest_server_t *server);

#endif /* HT_FED_FOREST_H */
