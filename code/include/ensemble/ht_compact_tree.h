/**
 * @file ht_compact_tree.h
 * @brief Lightweight, pointer-free inference-only decision tree (BFS-compacted).
 *
 * Shared foundation between Phase 9 (Micro-Ensembles) and Phase 11 (Fed-Forest).
 * Strips heavy learning state (stats_pool, node_drift, adwin_pool) and remaps
 * active topology indices densely into [0 .. num_nodes - 1] via iterative BFS.
 *
 * Memory footprint: ~1.32 KB for 64 nodes (float32), ~2.60 KB for 128 nodes.
 * Strictly C11 zero-malloc.
 */

#ifndef HT_COMPACT_TREE_H
#define HT_COMPACT_TREE_H

#include "ht_config.h"
#include "ht_memory.h"
#include "ht_tree.h"

#define HT_COMPACT_MAGIC        0x54435448U /**< ASCII "HTCT" in little-endian */
#define HT_COMPACT_VERSION      1U
#define HT_COMPACT_FLAG_DOUBLE  (1U << 0)
#define HT_COMPACT_FLAG_REG     (1U << 1)
#define HT_COMPACT_FLAG_SUBSPC  (1U << 2)
#define HT_COMPACT_FLAG_FIXED   (1U << 3) /**< Serialized with Q16.16 fixed-point ht_float_t */

/**
 * @brief Compact inference-only decision tree structure (zero stats_pool / drift).
 */
typedef struct {
    ht_node_t         nodes[HT_COMPACT_MAX_NODES];
    ht_float_t        train_weight_seen;
    ht_task_type_t    task_type;
    ht_feature_type_t feature_types[HT_MAX_FEATURES];
    uint16_t          num_nodes;
    uint16_t          root_idx;
    uint8_t           n_features;
    uint8_t           n_classes;
    uint8_t           has_subspace;
    uint8_t           feature_map[HT_MAX_FEATURES];
} ht_compact_tree_t;

/**
 * @brief Wire header for serialized compact trees.
 */
typedef struct {
    uint32_t          magic;
    uint16_t          version;
    uint16_t          flags;
    uint16_t          float_size;
    uint16_t          num_nodes;
    uint16_t          root_idx;
    uint8_t           n_features;
    uint8_t           n_classes;
    ht_float_t        train_weight_seen;
    ht_feature_type_t feature_types[HT_MAX_FEATURES];
    uint8_t           feature_map[HT_MAX_FEATURES];
} ht_compact_serial_header_t;

/**
 * @brief Compacts a full ht_tree_t into an inference-only ht_compact_tree_t via iterative BFS.
 *
 * Remaps reachable active nodes from src->root_idx into contiguous [0 .. num_nodes - 1].
 * If src contains more live nodes than HT_COMPACT_MAX_NODES, frontier internal nodes
 * are gracefully collapsed into leaves summarizing their subtrees (zero dynamic memory).
 *
 * @param dst Destination compact tree.
 * @param src Source full Hoeffding tree.
 * @return HT_OK on success, or negative error code.
 */
ht_status_t ht_compact_tree_from_tree(ht_compact_tree_t *dst, const ht_tree_t *src);

/**
 * @brief Optionally attaches a subspace feature projection map to a compact tree.
 *
 * When set, feature index f inside the compact tree reads x[feature_map[f]] from the
 * global input feature vector x.
 */
ht_status_t ht_compact_tree_set_subspace(ht_compact_tree_t *ct,
                                         const uint8_t *feature_map,
                                         uint8_t n_subspace_features);

/**
 * @brief Iterative root-to-leaf routing on a compact tree (handles NaN / missing values).
 */
uint16_t ht_compact_tree_route(const ht_compact_tree_t *ct,
                               const ht_value_t x[HT_MAX_FEATURES]);

/**
 * @brief Predicts class probability distribution (soft majority class at reached leaf).
 */
ht_status_t ht_compact_tree_predict_proba(const ht_compact_tree_t *ct,
                                          const ht_value_t x[HT_MAX_FEATURES],
                                          ht_float_t proba_out[HT_MAX_CLASSES]);

/**
 * @brief Predicts target class label (argmax of predicted probabilities).
 */
uint8_t ht_compact_tree_predict(const ht_compact_tree_t *ct,
                                const ht_value_t x[HT_MAX_FEATURES]);

/**
 * @brief Predicts continuous target value (mean target at reached leaf).
 */
ht_float_t ht_compact_tree_predict_reg(const ht_compact_tree_t *ct,
                                       const ht_value_t x[HT_MAX_FEATURES]);

/**
 * @brief Serializes compact tree to a contiguous binary buffer.
 *
 * Pass buffer == NULL to query the exact byte size in out_bytes_written.
 */
ht_status_t ht_compact_tree_save(const ht_compact_tree_t *ct,
                                 void *buffer,
                                 size_t buffer_cap,
                                 size_t *out_bytes_written);

/**
 * @brief Deserializes compact tree from a binary buffer.
 */
ht_status_t ht_compact_tree_load(ht_compact_tree_t *ct,
                                 const void *buffer,
                                 size_t buffer_len);

#endif /* HT_COMPACT_TREE_H */
