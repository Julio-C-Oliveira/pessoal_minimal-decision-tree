/**
 * @file ht_tree.h
 * @brief Complete Hoeffding Tree (VFDT) core in pure C11.
 *
 * Implements:
 * - Deterministic static layout fitting within <= 1 MB RAM.
 * - Iterative root-to-leaf routing with missing value handling (most_common_path).
 * - Multi-strategy leaf prediction: Majority Class (MC), Naive Bayes (NB with LSE),
 *   and Naive Bayes Adaptive (NBA tracking prequential MC vs NB performance).
 * - Online incremental learning cycle (ht_tree_learn_one) with pure distribution
 *   and max_share_to_split pre-checks, Hoeffding bound evaluation, and tie-breaking.
 * - In-place binary split with immediate recycling of parent leaf statistics.
 * - Memory management via leaf promise metric (Promise(l) = W_total * p_l * e_l)
 *   to deactivate less promising leaves when the stats pool is exhausted.
 * - Binary serialization (ht_tree_save / ht_tree_load) with relocatable uint16_t indices.
 *
 * Strictly adheres to C11 (-std=c11 -Wall -Wextra -Werror -pedantic) and zero malloc in runtime.
 */

#ifndef HT_TREE_H
#define HT_TREE_H

#include "ht_config.h"
#include "ht_memory.h"
#include "ht_observer.h"
#include "ht_criterion.h"
#include "ht_drift.h"

/**
 * @brief Supported leaf prediction strategies.
 */
typedef enum {
    HT_LEAF_MC  = 0, /**< Majority Class: predicts class distribution at leaf */
    HT_LEAF_NB  = 1, /**< Naive Bayes: computes joint likelihood with Log-Sum-Exp */
    HT_LEAF_NBA = 2  /**< Naive Bayes Adaptive: dynamically switches between MC and NB */
} ht_leaf_pred_t;

/**
 * @brief Configuration parameters for Hoeffding Tree induction.
 */
typedef struct {
    uint8_t             n_features;                     /**< Number of stream features */
    uint8_t             n_classes;                      /**< Number of target classes */
    ht_feature_type_t   feature_types[HT_MAX_FEATURES]; /**< Type per feature */
    ht_criterion_type_t criterion;                      /**< HT_CRIT_INFO_GAIN or HT_CRIT_GINI */
    ht_leaf_pred_t      leaf_prediction;                /**< MC, NB, or NBA */
    uint16_t            grace_period;                   /**< Samples between split attempts (default: 200) */
    uint16_t            max_depth;                      /**< Maximum tree depth allowed (default: 64) */
    ht_float_t          delta;                          /**< Split confidence parameter (default: 1e-7) */
    ht_float_t          tau;                            /**< Tie-breaking threshold (default: 0.05) */
    ht_float_t          min_branch_fraction;            /**< Min sample fraction per branch (default: 0.01) */
    ht_float_t          max_share_to_split;             /**< Majority class share limit (default: 0.99) */
    ht_float_t          nb_threshold;                   /**< Min weight before NB is allowed (default: 0.0) */
    bool                merit_preprune;                 /**< Include null split candidate (default: true) */
    bool                enable_hat;                     /**< Enable HAT concept drift adaptation */
    bool                bootstrap_sampling;             /**< Enable Poisson(1) online bootstrap sampling */
    uint16_t            drift_window_threshold;         /**< Min samples before alternate tree replacement (default: 300) */
    ht_float_t          switch_significance;            /**< Significance for alternate tree replacement (default: 0.05) */
    ht_float_t          adwin_delta;                    /**< Confidence parameter for ADWIN detectors (default: 0.002) */
} ht_config_params_t;

/**
 * @brief Node concept drift monitoring and alternate subtree metadata for HAT.
 */
typedef struct {
    ht_float_t mean_error;   /**< Online error rate tracked at this node */
    uint32_t   error_count;  /**< Number of error updates observed */
    uint16_t   adwin_idx;    /**< Index in ht_adwin_pool_t, or HT_INVALID_IDX */
    uint16_t   alt_tree_idx; /**< Root of alternate subtree in node_pool, or HT_INVALID_IDX */
} ht_node_drift_t;

/**
 * @brief Complete Hoeffding Tree structure.
 *
 * Contiguously bundles static memory pools, schema parameters, criterion vtable,
 * and prequential performance tracking counters within <= 1 MB RAM.
 */
typedef struct {
    ht_memory_pool_t             memory;
    ht_config_params_t           params;
    const ht_criterion_vtable_t *criterion_vtable;
    ht_adwin_pool_t             *adwin_pool;
    ht_node_drift_t              node_drift[HT_MAX_NODES];
    ht_float_t                   train_weight_seen;
    uint32_t                     rng_state;
    uint16_t                     root_idx;
    uint16_t                     n_active_leaves;
    uint16_t                     n_inactive_leaves;
    uint16_t                     n_decision_nodes;
    uint16_t                     max_depth_seen;
    uint16_t                     n_alternate_trees;
    uint16_t                     n_pruned_alternate_trees;
    uint16_t                     n_switch_alternate_trees;
    uint16_t                     pad;
} ht_tree_t;

/* Compile-time verification of strict memory budget */
_Static_assert(sizeof(ht_tree_t) <= (1024 * 1024), "ht_tree_t exceeds 1 MB RAM budget!");

/* Binary serialization definitions */
#define HT_SERIAL_MAGIC       0x54444656U  /**< ASCII "VFDT" in little-endian ('V','F','D','T') */
#define HT_SERIAL_VERSION     1U
#define HT_SERIAL_FLAG_DOUBLE (1U << 0)
#define HT_SERIAL_FLAG_STATS  (1U << 1)

/**
 * @brief Binary image header for serialized Hoeffding Tree models.
 */
typedef struct {
    uint32_t           magic;
    uint16_t           version;
    uint16_t           flags;
    uint16_t           float_size;
    uint16_t           num_nodes_in_use;
    uint16_t           num_stats_in_use;
    uint16_t           root_idx;
    uint16_t           n_active_leaves;
    uint16_t           n_inactive_leaves;
    uint16_t           n_decision_nodes;
    uint16_t           max_depth_seen;
    ht_config_params_t params;
    ht_float_t         train_weight_seen;
} ht_serial_header_t;

/* ========================================================================= */
/* --- Public API ---------------------------------------------------------- */
/* ========================================================================= */

/**
 * @brief Fills configuration parameters with standard VFDT defaults.
 *
 * @param params Output parameters struct.
 * @param n_features Number of stream features (1 .. HT_MAX_FEATURES).
 * @param n_classes Number of target classes (2 .. HT_MAX_CLASSES).
 */
void ht_default_params(ht_config_params_t *params, uint8_t n_features, uint8_t n_classes);

/**
 * @brief Initializes a Hoeffding Tree instance with the specified parameters.
 *
 * Clears memory, sets parameters, resolves criterion vtable, and creates root
 * as an active learning leaf.
 *
 * @param tree Pointer to tree instance.
 * @param params Pointer to configuration parameters.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_init(ht_tree_t *tree, const ht_config_params_t *params);

/**
 * @brief Iterative routing of a sample from root to destination leaf.
 *
 * Guaranteed loop termination with zero recursion on stack. In the presence
 * of missing values (NaN in numeric, or invalid category), follows the most
 * common path (most_common_path) based on accumulated subtree class weights.
 *
 * @param tree Pointer to tree instance.
 * @param x Input feature vector.
 * @param out_parent_idx Optional output pointer for parent node index (or HT_INVALID_IDX).
 * @param out_parent_branch Optional output pointer for parent branch taken (0 or 1).
 * @param out_depth Optional output pointer for leaf depth (root = 0).
 * @return Index of destination leaf node in node_pool, or HT_INVALID_IDX on error.
 */
uint16_t ht_tree_route(const ht_tree_t *tree,
                       const ht_value_t x[HT_MAX_FEATURES],
                       uint16_t *out_parent_idx,
                       uint8_t *out_parent_branch,
                       uint16_t *out_depth);

/**
 * @brief Computes predicted class probabilities for an input feature vector.
 *
 * Strictly follows configured leaf prediction policy:
 * - HT_LEAF_MC: normalized class distribution of the leaf.
 * - HT_LEAF_NB: Naive Bayes conditional likelihoods with Log-Sum-Exp.
 * - HT_LEAF_NBA: NB if leaf is active and nb_correct_weight >= mc_correct_weight, else MC.
 * Inactive leaves always predict using Majority Class.
 *
 * @param tree Pointer to tree instance.
 * @param x Input feature vector.
 * @param proba_out Output probability array of size HT_MAX_CLASSES.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_predict_proba(const ht_tree_t *tree,
                                  const ht_value_t x[HT_MAX_FEATURES],
                                  ht_float_t proba_out[HT_MAX_CLASSES]);

/**
 * @brief Predicts the target class index for an input feature vector (argmax).
 *
 * @param tree Pointer to tree instance.
 * @param x Input feature vector.
 * @return Predicted class index (0 .. n_classes - 1).
 */
uint8_t ht_tree_predict(const ht_tree_t *tree,
                        const ht_value_t x[HT_MAX_FEATURES]);

/**
 * @brief Online incremental learning step with a single labeled sample.
 *
 * Routes sample to destination leaf, updates prequential accuracy counters
 * (for NBA), updates leaf statistics, and evaluates split conditions when
 * grace_period threshold is reached.
 *
 * @param tree Pointer to tree instance.
 * @param x Input feature vector.
 * @param y True class label (0 .. n_classes - 1).
 * @param weight Sample weight (> 0.0).
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_learn_one(ht_tree_t *tree,
                              const ht_value_t x[HT_MAX_FEATURES],
                              uint8_t y,
                              ht_float_t weight);

/**
 * @brief Computes the Promise metric for a node: Promise(l) = W_l - max_c(w_{l,c}).
 *
 * Represents the upper bound on error reduction from splitting leaf l.
 *
 * @param node Pointer to structural node.
 * @param n_classes Number of target classes.
 * @return Promise value (>= 0.0).
 */
ht_float_t ht_tree_node_promise(const ht_node_t *node, uint8_t n_classes);

/**
 * @brief Deactivates an active leaf, returning its statistics block to the pool.
 *
 * The leaf remains in the tree and predicts using Majority Class.
 *
 * @param tree Pointer to tree instance.
 * @param leaf_idx Index of leaf to deactivate.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_deactivate_leaf(ht_tree_t *tree, uint16_t leaf_idx);

/**
 * @brief Reactivates an inactive leaf if stats pool space is available.
 *
 * @param tree Pointer to tree instance.
 * @param leaf_idx Index of leaf to reactivate.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_reactivate_leaf(ht_tree_t *tree, uint16_t leaf_idx);

/**
 * @brief Deactivates the active leaf with the lowest promise across the tree.
 *
 * @param tree Pointer to tree instance.
 * @return HT_OK on success, or HT_ERR_OUT_OF_STATS if no active leaves exist.
 */
ht_status_t ht_tree_deactivate_least_promising_leaf(ht_tree_t *tree);

/**
 * @brief Serializes the tree model to a contiguous binary buffer.
 *
 * When buffer is NULL, returns required byte capacity in out_bytes_written.
 *
 * @param tree Pointer to tree instance.
 * @param buffer Destination buffer.
 * @param buffer_cap Destination buffer capacity in bytes.
 * @param out_bytes_written Output pointer for number of bytes written.
 * @param include_training_stats true to include leaf observers, false for inference-only.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_save(const ht_tree_t *tree,
                         void *buffer,
                         size_t buffer_cap,
                         size_t *out_bytes_written,
                         bool include_training_stats);

/**
 * @brief Deserializes a tree model from a binary buffer.
 *
 * Validates header, magic number, format version, and float precision.
 * Reconstructs memory pool free-lists in O(N).
 *
 * @param tree Output tree instance.
 * @param buffer Source binary buffer.
 * @param buffer_len Length of source buffer in bytes.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_load(ht_tree_t *tree,
                         const void *buffer,
                         size_t buffer_len);

/* --- HAT (Hoeffding Adaptive Tree) Public API --- */

/**
 * @brief Fills configuration parameters with standard HAT defaults matching River.
 *
 * @param params Output parameters struct.
 * @param n_features Number of stream features.
 * @param n_classes Number of target classes.
 */
void ht_default_hat_params(ht_config_params_t *params, uint8_t n_features, uint8_t n_classes);

/**
 * @brief Initializes a Hoeffding Adaptive Tree (HAT) instance with an external ADWIN pool.
 *
 * @param tree Pointer to tree instance.
 * @param params Pointer to configuration parameters.
 * @param adwin_pool Pointer to initialized ADWIN detector pool.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_init_hat(ht_tree_t *tree,
                             const ht_config_params_t *params,
                             ht_adwin_pool_t *adwin_pool);

/**
 * @brief Attaches (or detaches) an ADWIN detector pool to a tree instance.
 *
 * @param tree Pointer to tree instance.
 * @param adwin_pool Pointer to pool structure (or NULL to disable HAT adaptation).
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_set_adwin_pool(ht_tree_t *tree, ht_adwin_pool_t *adwin_pool);

/**
 * @brief Recursively/iteratively frees an entire subtree and returns all its nodes,
 *        leaf stats blocks, and ADWIN detectors back to their static pools.
 *
 * Guaranteed loop termination with zero recursion on stack.
 *
 * @param tree Pointer to tree instance.
 * @param subtree_root_idx Root node index of subtree to dismantle.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_tree_free_subtree(ht_tree_t *tree, uint16_t subtree_root_idx);

#endif /* HT_TREE_H */
