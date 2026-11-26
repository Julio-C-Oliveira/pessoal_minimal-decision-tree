/**
 * @file ht_fed_vfdt.h
 * @brief Mode 2 Federated Learning: Fed-VFDT (Global Synchronized Tree) in pure C11.
 *
 * Supports two configurable aggregation modes over a single topology-synchronized tree:
 * - HT_FED_VFDT_COORDINATED: Canonical Fed-VFDT (Silva, Vinagre & Gama, IEEE ICTAI)
 *   where clients transmit ultra-compact split proposals and the server decides splits
 *   via Quorum, Majority Vote, or Merit-Weighted consensus.
 * - HT_FED_VFDT_STATS_MERGE: Exact Sufficient Statistics Merge using Chan/Welford,
 *   Categorical summation, Ben-Haim histogram fusion, Regression slot fusion, and
 *   sample-weighted FedAvg for linear SGD leaves, with stateless round re-merge
 *   (zero double-counting, zero extra delta buffers on clients).
 *
 * Includes topology_version tracking, circular split_log incremental catch-up for
 * stragglers, automatic full-snapshot fallback, and server-coordinated drift reset.
 */

#ifndef HT_FED_VFDT_H
#define HT_FED_VFDT_H

#include "ht_config.h"
#include "ht_tree.h"
#include "ht_fed_proto.h"

#ifndef HT_FED_SERVER_MAX_PROPOSALS
#define HT_FED_SERVER_MAX_PROPOSALS 64U
#endif

/**
 * @brief Fed-VFDT aggregation mode.
 */
typedef enum {
    HT_FED_VFDT_COORDINATED = 0, /**< Canonical Fed-VFDT: compact split proposals + consensus */
    HT_FED_VFDT_STATS_MERGE = 1  /**< Exact sufficient statistics merge + global Hoeffding bound */
} ht_fed_vfdt_mode_t;

/**
 * @brief Consensus rule used by the server in HT_FED_VFDT_COORDINATED mode.
 */
typedef enum {
    HT_FED_CONSENSUS_QUORUM         = 0, /**< Per-feature quorum (feat_votes[f] >= min_quorum), highest merit sum wins */
    HT_FED_CONSENSUS_MAJORITY       = 1, /**< Plurality vote (most frequent feature), tie-break by highest merit */
    HT_FED_CONSENSUS_MERIT          = 2, /**< Best individual merit max(G_k) across >= min_quorum proposals (paper faithful) */
    HT_FED_CONSENSUS_WEIGHTED_MERIT = 3  /**< Highest sample-weighted merit sum sum(w_k * G_k) across >= min_quorum proposals */
} ht_fed_vfdt_consensus_t;

/**
 * @brief Numeric/categorical split threshold aggregation rule in COORDINATED mode.
 */
typedef enum {
    HT_FED_THRESH_MAX_MERIT     = 0, /**< Exact threshold/category from highest-merit winning proposal (Fed-VFDT default) */
    HT_FED_THRESH_WEIGHTED_MEAN = 1  /**< Sample-weighted mean threshold / weighted mode category across winning proposals */
} ht_fed_vfdt_thresh_agg_t;

/**
 * @brief Downlink topology command types recorded in split_log.
 */
typedef enum {
    HT_FED_CMD_SPLIT         = 0, /**< Deterministic binary split on leaf node_idx */
    HT_FED_CMD_RESET_SUBTREE = 1  /**< Coordinated concept-drift reset of subtree node_idx to active leaf */
} ht_fed_cmd_type_t;

/**
 * @brief One versioned global topology command stored in server split_log and sent downlink.
 */
typedef struct {
    ht_split_candidate_t cand;
    uint32_t             version;
    uint16_t             node_idx;
    uint16_t             depth;
    uint8_t              cmd_type;
    uint8_t              reserved[3];
} ht_fed_split_cmd_t;

/**
 * @brief Compact per-leaf split proposal sent uplink in HT_FED_VFDT_COORDINATED mode.
 */
typedef struct {
    ht_split_candidate_t best;
    ht_float_t           total_weight;
    ht_float_t           second_best_merit;
    uint16_t             leaf_idx;
    uint16_t             depth;
    uint8_t              locally_ready;
    uint8_t              reserved[3];
} ht_fed_split_proposal_t;

/**
 * @brief Client configuration for Fed-VFDT.
 */
typedef struct {
    ht_config_params_t tree_params;
    ht_fed_vfdt_mode_t mode;
    uint16_t           client_id;
} ht_fed_vfdt_client_cfg_t;

/**
 * @brief Embedded Fed-VFDT client instance.
 */
typedef struct {
    ht_tree_t          tree;
    ht_fed_vfdt_mode_t mode;
    uint32_t           topology_version;
    uint32_t           round_id;
    uint16_t           client_id;
    uint16_t           drift_alert_node; /**< HT_INVALID_IDX unless local drift is flagged */
} ht_fed_vfdt_client_t;

/**
 * @brief Server configuration for Fed-VFDT.
 */
typedef struct {
    ht_config_params_t       tree_params;
    ht_fed_vfdt_mode_t       mode;
    ht_fed_vfdt_consensus_t  consensus;
    ht_fed_vfdt_thresh_agg_t threshold_agg;        /**< HT_FED_THRESH_MAX_MERIT (default) or HT_FED_THRESH_WEIGHTED_MEAN */
    uint16_t                 min_quorum;           /**< Minimum agreeing (QUORUM) or participating proposals (default: 2) */
    uint16_t                 min_drift_quorum;     /**< Minimum client drift alerts to trigger reset (default: 2) */
    uint16_t                 max_splits_per_round; /**< 0 = unlimited independent leaves/round (default), 1 = paper 1-split/round */
    bool                     require_locally_ready;/**< true (default, paper faithful): only count Hoeffding-bound-ready proposals */
} ht_fed_vfdt_server_cfg_t;

/**
 * @brief Static zero-malloc Fed-VFDT coordinating server (single global tree + logs).
 */
typedef struct {
    ht_tree_t                global_tree;
    ht_fed_vfdt_server_cfg_t cfg;
    ht_fed_split_cmd_t       split_log[HT_FED_SPLIT_LOG_CAP];
    ht_fed_split_proposal_t  round_proposals[HT_FED_SERVER_MAX_PROPOSALS];
    union {
        ht_float_t           class_counts[HT_MAX_CLASSES];
        ht_var_stat_t        reg_stats;
    } merge_leaf_delta[HT_MAX_ACTIVE_LEAVES];
    ht_float_t               merge_sgd_weight_sum[HT_MAX_ACTIVE_LEAVES];
    uint16_t                 drift_vote_nodes[HT_FED_MAX_CLIENTS];
    uint16_t                 drift_vote_counts[HT_FED_MAX_CLIENTS];
    uint32_t                 topology_version;
    uint32_t                 split_log_head_version;
    uint32_t                 round_id;
    uint16_t                 n_round_proposals;
    uint16_t                 n_round_clients;
    uint16_t                 n_drift_entries;
    uint16_t                 n_stale_rejected;
} ht_fed_vfdt_server_t;

/* --- Client API ---------------------------------------------------------- */

void ht_fed_vfdt_default_client_config(ht_fed_vfdt_client_cfg_t *cfg,
                                       uint16_t client_id,
                                       ht_fed_vfdt_mode_t mode,
                                       const ht_config_params_t *tree_params);

ht_status_t ht_fed_vfdt_client_init(ht_fed_vfdt_client_t *client,
                                    const ht_fed_vfdt_client_cfg_t *cfg);

ht_status_t ht_fed_vfdt_client_predict_proba(const ht_fed_vfdt_client_t *client,
                                             const ht_value_t x[HT_MAX_FEATURES],
                                             ht_float_t proba_out[HT_MAX_CLASSES]);

uint8_t ht_fed_vfdt_client_predict(const ht_fed_vfdt_client_t *client,
                                   const ht_value_t x[HT_MAX_FEATURES]);

ht_status_t ht_fed_vfdt_client_learn_one(ht_fed_vfdt_client_t *client,
                                         const ht_value_t x[HT_MAX_FEATURES],
                                         uint8_t y,
                                         ht_float_t weight);

ht_float_t ht_fed_vfdt_client_predict_reg(const ht_fed_vfdt_client_t *client,
                                          const ht_value_t x[HT_MAX_FEATURES]);

ht_status_t ht_fed_vfdt_client_learn_one_reg(ht_fed_vfdt_client_t *client,
                                             const ht_value_t x[HT_MAX_FEATURES],
                                             ht_float_t y,
                                             ht_float_t weight);

/**
 * @brief Flags a local concept-drift alert on node_idx (e.g. client->tree.root_idx)
 *        to be transmitted in the next uplink update.
 */
void ht_fed_vfdt_client_flag_drift(ht_fed_vfdt_client_t *client, uint16_t node_idx);

/**
 * @brief Exports client uplink packet (SPLIT_PROPOSAL or LEAF_STATS depending on mode).
 */
ht_status_t ht_fed_vfdt_client_export_update(ht_fed_vfdt_client_t *client,
                                             void *out_buf,
                                             size_t buf_cap,
                                             size_t *out_written);

/**
 * @brief Applies a server downlink packet (SPLIT_BROADCAST catch-up or FULL_SNAPSHOT).
 */
ht_status_t ht_fed_vfdt_client_apply_update(ht_fed_vfdt_client_t *client,
                                            const void *buf,
                                            size_t buf_len);

/* --- Server API ---------------------------------------------------------- */

void ht_fed_vfdt_default_server_config(ht_fed_vfdt_server_cfg_t *cfg,
                                       ht_fed_vfdt_mode_t mode,
                                       const ht_config_params_t *tree_params);

ht_status_t ht_fed_vfdt_server_init(ht_fed_vfdt_server_t *server,
                                    const ht_fed_vfdt_server_cfg_t *cfg);

/**
 * @brief Starts a new aggregation round, resetting stateless round accumulators.
 */
ht_status_t ht_fed_vfdt_server_begin_round(ht_fed_vfdt_server_t *server);

/**
 * @brief Ingests a client uplink packet (SPLIT_PROPOSAL or LEAF_STATS).
 *
 * Returns HT_ERR_STALE_VERSION if client's topology_version != server->topology_version
 * (caller should then invoke ht_fed_vfdt_server_export_update with client's version to resync).
 */
ht_status_t ht_fed_vfdt_server_ingest(ht_fed_vfdt_server_t *server,
                                      const void *buf,
                                      size_t buf_len);

/**
 * @brief Finalizes the round: executes coordinated drift resets and global splits,
 *        recording all topology changes in split_log and advancing topology_version.
 *
 * @param server Pointer to server instance.
 * @param out_cmds_applied Optional output count of splits/resets executed this round.
 */
ht_status_t ht_fed_vfdt_server_end_round(ht_fed_vfdt_server_t *server,
                                         uint16_t *out_cmds_applied);

/**
 * @brief Exports a downlink update tailored to a client's current topology_version.
 *
 * Emits an incremental SPLIT_BROADCAST (from split_log) whenever possible, or
 * automatically falls back to FULL_SNAPSHOT if the client fell behind by more than
 * HT_FED_SPLIT_LOG_CAP versions.
 */
ht_status_t ht_fed_vfdt_server_export_update(const ht_fed_vfdt_server_t *server,
                                             uint16_t target_client_id,
                                             uint32_t client_topology_version,
                                             void *out_buf,
                                             size_t buf_cap,
                                             size_t *out_written);

#endif /* HT_FED_VFDT_H */
