/**
 * @file ht_memory.h
 * @brief Deterministic Slab / Pool Allocator for the Embedded Hoeffding Tree (VFDT / HAT).
 *
 * Implements two independent static pools (node_pool and stats_pool) with
 * O(1) free-list index stacks. Strictly prohibits any dynamic allocation
 * (malloc/free) in the learning and inference paths, guaranteeing predictable
 * execution time and strict adherence to the <= 1 MB RAM budget.
 */

#ifndef HT_MEMORY_H
#define HT_MEMORY_H

#include "ht_config.h"
#include <string.h>

/* Node structural flags */
#define HT_NODE_IS_LEAF      (1U << 0)
#define HT_NODE_SPLIT_CAT    (1U << 1)
#define HT_NODE_IN_USE       (1U << 7)

/**
 * @brief Sufficient variance statistics for continuous regression target y.
 *
 * Implements West/Welford incremental variance and Chan's parallel formulas.
 * Exact size: 3 floats (12 bytes in float32 / 24 bytes in float64).
 */
typedef struct {
    ht_float_t count;      /**< Total accumulated weight n */
    ht_float_t mean;       /**< Incremental mean \mu_y */
    ht_float_t m2;         /**< Sum of squared differences from mean S_y (M2) */
} ht_var_stat_t;

/**
 * @brief Quantization slot for continuous numeric feature in regression.
 *
 * Tracks the incremental mean of attribute x and variance statistics of target y.
 * Exact size: 4 floats (16 bytes in float32 / 32 bytes in float64).
 */
typedef struct {
    ht_float_t    x_mean;  /**< Incremental mean of feature x in this slot */
    ht_var_stat_t y_stats; /**< Sufficient variance statistics (n, \mu_y, S_y) of target y */
} ht_reg_num_slot_t;

/**
 * @brief Numeric feature observer for regression using HT_REG_NUM_SLOTS ordered slots.
 *
 * Exact size: 5 * 16 = 80 bytes in float32 / 5 * 32 = 160 bytes in float64.
 */
typedef struct {
    ht_reg_num_slot_t slots[HT_REG_NUM_SLOTS];
} ht_reg_numeric_obs_t;

/**
 * @brief Categorical nominal observer for regression.
 *
 * Maintains target variance statistics for up to HT_MAX_CATEGORIES.
 * Exact size: 5 * 12 = 60 bytes in float32 / 5 * 24 = 120 bytes in float64.
 */
typedef struct {
    ht_var_stat_t cat_stats[HT_MAX_CATEGORIES];
} ht_reg_categorical_obs_t;

#define HT_NODE_FLAGS_OFFSET (sizeof(ht_float_t) * (HT_MAX_CLASSES >= 4 ? HT_MAX_CLASSES : 4) + sizeof(uint16_t))

/**
 * @brief Permanent structural node in the decision tree (asymmetric leaf vs branch union).
 *
 * Implements Phase 7 Hypothesis 2 (H2): Asymmetric Topology Union.
 * In a Hoeffding tree, a node is strictly an active/inactive LEAF or an internal BRANCH,
 * never both simultaneously:
 * - LEAF: class_counts[HT_MAX_CLASSES] (or reg_stats) + stats_idx.
 * - BRANCH: split_val (threshold / cat_val) + left_child + right_child + feature_idx (9-13 bytes).
 * - INACTIVE: next_free intrusive free-list link (2 bytes).
 * - HEADER: flags and reserved placed at offset HT_NODE_FLAGS_OFFSET, ensuring
 *   flags never collides with leaf or branch payload regardless of HT_MAX_CLASSES macro.
 *
 * Sized at exactly 20 bytes in float32 and 40 bytes in float64 with default HT_MAX_CLASSES=4.
 */
typedef struct {
    union {
        /* Leaf node state */
        struct {
            union {
                ht_float_t    class_counts[HT_MAX_CLASSES];
                ht_var_stat_t reg_stats;
            };
            uint16_t stats_idx;
        };
        /* Branch (internal decision) node state */
        struct {
            union {
                ht_float_t threshold;
                int32_t    cat_val;
            } split_val;
            uint16_t left_child;
            uint16_t right_child;
            uint16_t drift_idx;  /**< Dedicated slot in ht_drift_pool_t (0..127), or HT_INVALID_IDX */
            uint8_t  feature_idx;
        };
        /* Free list link when node is inactive */
        uint16_t next_free;
        /* Common node header at fixed offset ensuring flags never collides with payload */
        struct {
            uint8_t _pad_flags[HT_NODE_FLAGS_OFFSET];
            uint8_t flags;
            uint8_t reserved;
        };
    };
} ht_node_t;

/**
 * @brief Gaussian Welford statistics for 1 class of a numeric feature.
 *
 * Maintains 5 incremental scalars: count, mean, centered sum of squares (m2),
 * and observed class bounds [min_val, max_val] for exact River-equivalent split merit evaluation.
 * Sized at exactly 20 bytes in float32 and 40 bytes in float64.
 */
typedef struct {
    ht_float_t count;      /* Accumulated weight n_c */
    ht_float_t mean;       /* Incremental mean \mu_c */
    ht_float_t m2;         /* Centered sum of squared differences M_{2,c} */
    ht_float_t min_val;    /* Minimum observed value for class c */
    ht_float_t max_val;    /* Maximum observed value for class c */
} ht_gaussian_class_stat_t;

/**
 * @brief Gaussian observer across all classes for 1 numeric feature.
 *
 * Sized at exactly 80 bytes in float32 and 160 bytes in float64 across HT_MAX_CLASSES=4.
 */
typedef struct {
    ht_gaussian_class_stat_t classes[HT_MAX_CLASSES];
} ht_gaussian_obs_t;

typedef ht_gaussian_obs_t ht_gaussian_feature_obs_t;

/**
 * @brief Categorical nominal observer for 1 feature across all classes.
 *
 * Maintains frequency counts matrix for each category in each class.
 */
typedef struct {
    ht_float_t counts[HT_MAX_CLASSES][HT_MAX_CATEGORIES];
} ht_categorical_obs_t;

/**
 * @brief Multi-class Ben-Haim & Tom-Tov dynamic histogram bin.
 *
 * Exact size: (1 + 4) * 4 = 20 bytes in float32 / (1 + 4) * 8 = 40 bytes in float64.
 */
typedef struct {
    ht_float_t center;
    ht_float_t counts[HT_MAX_CLASSES];
} ht_hist_bin_t;

/**
 * @brief Non-parametric dynamic histogram observer for classification.
 *
 * Exact size: 4 * 20 = 80 bytes in float32 / 4 * 40 = 160 bytes in float64.
 */
typedef struct {
    ht_hist_bin_t bins[HT_HIST_MAX_BINS];
} ht_hist_obs_t;

/**
 * @brief Unified feature observer representation (no vtable).
 *
 * Exactly 80 bytes in float32 and 160 bytes in float64 across all 5 modes.
 */
typedef union {
    ht_gaussian_obs_t        num;
    ht_hist_obs_t            hist;
    ht_categorical_obs_t     cat;
    ht_reg_numeric_obs_t     reg_num;
    ht_reg_categorical_obs_t reg_cat;
} ht_feature_obs_t;

/**
 * @brief Heavy learning statistics allocated only for ACTIVE leaves.
 *
 * Holds split attempt timestamp, NBA accuracy weights / regression fMSE accumulators,
 * SGD linear model weights/intercept, and observers for all features.
 * Recycled immediately on node split.
 * Sized at exactly 1360 bytes in float32 and 2720 bytes in float64.
 */
typedef struct {
    ht_float_t last_split_attempt_at;
    union {
        /* Classification metadata: NBA adaptive weights (8 bytes in float32 / 16 bytes in double) */
        struct {
            ht_float_t mc_correct_weight;
            ht_float_t nb_correct_weight;
        };
        /* Regression metadata: fMSE accumulators and SGD linear weights (76 bytes in float32 / 152 bytes in double) */
        struct {
            ht_float_t fmse_mean;
            ht_float_t fmse_model;
            ht_float_t reg_weights[HT_MAX_FEATURES];
            ht_float_t reg_intercept;
        };
        /* Intrusive free-list next index when stats block is free */
        uint16_t next_free;
    };
    ht_feature_obs_t observers[HT_MAX_FEATURES];
} ht_leaf_stats_t;

/**
 * @brief Deterministic slab/pool allocator managing node and leaf statistics pools.
 *
 * Employs intrusive free lists directly inside inactive node/stats buffers, eliminating
 * external index arrays (free_nodes[] and free_stats[]) to save 8.7 KB of SRAM.
 */
typedef struct {
    ht_node_t       node_pool[HT_MAX_NODES];
    ht_leaf_stats_t stats_pool[HT_MAX_ACTIVE_LEAVES];
    uint16_t        free_nodes_head; /**< Head index of intrusive free-nodes list */
    uint16_t        free_stats_head; /**< Head index of intrusive free-stats list */
    uint16_t        nodes_allocated; /**< Number of currently active nodes */
    uint16_t        stats_allocated; /**< Number of currently active leaf stats blocks */
} ht_memory_pool_t;

/* Compile-time static assertions for exact memory layout in default configuration */
#if HT_MAX_CLASSES == 4 && HT_MAX_FEATURES == 16 && HT_REG_NUM_SLOTS == 5
#ifndef HT_USE_DOUBLE
_Static_assert(sizeof(ht_node_t) == 20, "ht_node_t must be exactly 20 bytes in float32 mode (H2 asymmetric union)");
_Static_assert(sizeof(ht_gaussian_class_stat_t) == 20, "ht_gaussian_class_stat_t must be 20 bytes in float32 mode");
_Static_assert(sizeof(ht_gaussian_obs_t) == 80, "ht_gaussian_obs_t must be 80 bytes in float32 mode");
_Static_assert(sizeof(ht_categorical_obs_t) == 80, "ht_categorical_obs_t must be 80 bytes in float32 mode");
_Static_assert(sizeof(ht_var_stat_t) == 12, "ht_var_stat_t must be exactly 12 bytes in float32 mode");
_Static_assert(sizeof(ht_reg_num_slot_t) == 16, "ht_reg_num_slot_t must be exactly 16 bytes in float32 mode");
_Static_assert(sizeof(ht_reg_numeric_obs_t) == 80, "ht_reg_numeric_obs_t must be exactly 80 bytes in float32 mode");
_Static_assert(sizeof(ht_reg_categorical_obs_t) == 60, "ht_reg_categorical_obs_t must be exactly 60 bytes in float32 mode");
_Static_assert(sizeof(ht_hist_bin_t) == 20, "ht_hist_bin_t must be exactly 20 bytes in float32 mode");
_Static_assert(sizeof(ht_hist_obs_t) == 80, "ht_hist_obs_t must be exactly 80 bytes in float32 mode");
_Static_assert(sizeof(ht_feature_obs_t) == 80, "ht_feature_obs_t must be 80 bytes in float32 mode");
_Static_assert(sizeof(ht_leaf_stats_t) == 1360, "ht_leaf_stats_t must be exactly 1360 bytes in float32 mode");
#else
_Static_assert(sizeof(ht_node_t) == 40, "ht_node_t must be exactly 40 bytes in float64 mode (H2 asymmetric union)");
_Static_assert(sizeof(ht_gaussian_class_stat_t) == 40, "ht_gaussian_class_stat_t must be 40 bytes in float64 mode");
_Static_assert(sizeof(ht_gaussian_obs_t) == 160, "ht_gaussian_obs_t must be 160 bytes in float64 mode");
_Static_assert(sizeof(ht_categorical_obs_t) == 160, "ht_categorical_obs_t must be 160 bytes in float64 mode");
_Static_assert(sizeof(ht_var_stat_t) == 24, "ht_var_stat_t must be exactly 24 bytes in float64 mode");
_Static_assert(sizeof(ht_reg_num_slot_t) == 32, "ht_reg_num_slot_t must be exactly 32 bytes in float64 mode");
_Static_assert(sizeof(ht_reg_numeric_obs_t) == 160, "ht_reg_numeric_obs_t must be exactly 160 bytes in float64 mode");
_Static_assert(sizeof(ht_reg_categorical_obs_t) == 120, "ht_reg_categorical_obs_t must be exactly 120 bytes in float64 mode");
_Static_assert(sizeof(ht_hist_bin_t) == 40, "ht_hist_bin_t must be exactly 40 bytes in float64 mode");
_Static_assert(sizeof(ht_hist_obs_t) == 160, "ht_hist_obs_t must be exactly 160 bytes in float64 mode");
_Static_assert(sizeof(ht_feature_obs_t) == 160, "ht_feature_obs_t must be 160 bytes in float64 mode");
_Static_assert(sizeof(ht_leaf_stats_t) == 2720, "ht_leaf_stats_t must be exactly 2720 bytes in float64 mode");
#endif
#endif
#if HT_MAX_CLASSES == 4 && HT_MAX_FEATURES == 16 && HT_REG_NUM_SLOTS == 5 && HT_MAX_NODES == 4096 && HT_MAX_ACTIVE_LEAVES == 256
#ifndef HT_USE_DOUBLE
_Static_assert(sizeof(ht_memory_pool_t) == 430088, "ht_memory_pool_t must be exactly 430088 bytes in float32 mode (H2 node union)");
#else
_Static_assert(sizeof(ht_memory_pool_t) == 860168, "ht_memory_pool_t must be exactly 860168 bytes in float64 mode (H2 node union)");
#endif
#endif
_Static_assert(sizeof(ht_memory_pool_t) <= (1024 * 1024), "ht_memory_pool_t exceeds 1 MB RAM budget!");

/**
 * @brief Initializes memory pool free lists and zero-initializes memory.
 *
 * Sets up free-index stacks for O(1) allocation order (first alloc returns index 0).
 *
 * @param pool Pointer to the memory pool.
 */
void ht_memory_init(ht_memory_pool_t *pool);

/**
 * @brief Allocates an uninitialized node from the node pool in O(1).
 *
 * Initializes default structural values:
 * - left_child = right_child = stats_idx = HT_INVALID_IDX
 * - feature_idx = HT_NO_FEATURE
 * - flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE
 * - class_counts zeroed out
 *
 * @param pool Pointer to the memory pool.
 * @return Index of allocated node, or HT_INVALID_IDX if pool is exhausted.
 */
uint16_t ht_node_alloc(ht_memory_pool_t *pool);

/**
 * @brief Frees a node index back to the free list in O(1).
 *
 * Clears flags and resets child references to prevent dangling pointers.
 *
 * @param pool Pointer to the memory pool.
 * @param node_idx Index of node to free.
 */
void ht_node_free(ht_memory_pool_t *pool, uint16_t node_idx);

/**
 * @brief Allocates an uninitialized leaf statistics block in O(1).
 *
 * Zero-initializes all observers and metadata via memset.
 *
 * @param pool Pointer to the memory pool.
 * @return Index of allocated leaf stats, or HT_INVALID_IDX if pool is exhausted.
 */
uint16_t ht_stats_alloc(ht_memory_pool_t *pool);

/**
 * @brief Frees a leaf statistics index back to the free list in O(1).
 *
 * Zeroes out the block to ensure fresh state upon reallocation.
 *
 * @param pool Pointer to the memory pool.
 * @param stats_idx Index of leaf statistics block to free.
 */
void ht_stats_free(ht_memory_pool_t *pool, uint16_t stats_idx);

/**
 * @brief Calculates the exact number of bytes currently active in the pool.
 *
 * @param pool Pointer to the memory pool.
 * @return Active bytes used (nodes_allocated * sizeof(ht_node_t) + stats_allocated * sizeof(ht_leaf_stats_t)).
 */
size_t ht_memory_bytes_used(const ht_memory_pool_t *pool);

/**
 * @brief Returns total static byte capacity of the memory pool struct.
 *
 * @return Total size of ht_memory_pool_t in bytes (must be <= 1 MB).
 */
size_t ht_memory_total_capacity_bytes(void);

/**
 * @brief Direct pointer accessor for a node given its index with bounds checking.
 *
 * @param pool Pointer to the memory pool.
 * @param idx Node index.
 * @return Pointer to ht_node_t, or NULL if index is invalid / out of bounds.
 */
ht_node_t *ht_node_get(ht_memory_pool_t *pool, uint16_t idx);

/**
 * @brief Read-only pointer accessor for a node with bounds checking.
 */
const ht_node_t *ht_node_get_const(const ht_memory_pool_t *pool, uint16_t idx);

/**
 * @brief Direct pointer accessor for leaf statistics given its index with bounds checking.
 *
 * @param pool Pointer to the memory pool.
 * @param idx Leaf statistics index.
 * @return Pointer to ht_leaf_stats_t, or NULL if index is invalid / out of bounds.
 */
ht_leaf_stats_t *ht_stats_get(ht_memory_pool_t *pool, uint16_t idx);

/**
 * @brief Read-only pointer accessor for leaf statistics with bounds checking.
 */
const ht_leaf_stats_t *ht_stats_get_const(const ht_memory_pool_t *pool, uint16_t idx);

/**
 * @brief Returns number of available free nodes.
 */
uint16_t ht_node_available(const ht_memory_pool_t *pool);

/**
 * @brief Returns number of available free leaf stats blocks.
 */
uint16_t ht_stats_available(const ht_memory_pool_t *pool);

/**
 * @brief Returns number of currently allocated nodes.
 */
uint16_t ht_node_count(const ht_memory_pool_t *pool);

/**
 * @brief Returns number of currently allocated leaf stats blocks.
 */
uint16_t ht_stats_count(const ht_memory_pool_t *pool);

#endif /* HT_MEMORY_H */
