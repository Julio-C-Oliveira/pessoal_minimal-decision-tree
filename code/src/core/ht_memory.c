/**
 * @file ht_memory.c
 * @brief Implementation of the Deterministic Slab / Pool Allocator for Hoeffding Tree.
 */

#include "ht_memory.h"

bool ht_memory_init_stride(ht_memory_pool_t *pool, uint16_t n_features, ht_task_type_t task_type) {
    if (!pool) {
        return false;
    }
#ifdef HT_CLASSIF_ONLY
    if (task_type == HT_TASK_REGRESSION) {
        return false;
    }
#endif
    if (n_features == 0 || n_features > HT_MAX_FEATURES) {
        return false;
    }

    memset(pool, 0, sizeof(*pool));

    /* Initialize node intrusive free-list in descending order so index 0 is at head */
    pool->free_nodes_head = HT_INVALID_IDX;
    for (int32_t i = (int32_t)HT_MAX_NODES - 1; i >= 0; --i) {
        pool->node_pool[i].flags = 0;
        pool->node_pool[i].next_free = pool->free_nodes_head;
        pool->free_nodes_head = (uint16_t)i;
    }
    pool->nodes_allocated = 0;

    /* Calculate dynamic leaf stride */
    size_t header_bytes = offsetof(ht_leaf_stats_t, observers);
    size_t obs_bytes = (size_t)n_features * sizeof(ht_feature_obs_t);
    size_t reg_bytes = (task_type == HT_TASK_REGRESSION) ? ((size_t)n_features * sizeof(ht_float_t)) : 0;
    size_t raw_bytes = header_bytes + obs_bytes + reg_bytes;

    size_t align_mask = sizeof(ht_float_t) - 1;
    size_t stride = (raw_bytes + align_mask) & ~align_mask;

    if (stride == 0 || stride > HT_STATS_STORAGE_BYTES) {
        return false;
    }

    size_t raw_cap = HT_STATS_STORAGE_BYTES / stride;
    if (raw_cap == 0) {
        return false;
    }

    uint16_t max_leaves = (raw_cap > (size_t)HT_MAX_ACTIVE_LEAVES)
        ? (uint16_t)HT_MAX_ACTIVE_LEAVES
        : (uint16_t)raw_cap;

    pool->leaf_stride = (uint16_t)stride;
    pool->max_leaves_capacity = max_leaves;

    /* Initialize stats intrusive free-list in descending order so index 0 is at head */
    pool->free_stats_head = HT_INVALID_IDX;
    for (int32_t i = (int32_t)max_leaves - 1; i >= 0; --i) {
        ht_leaf_stats_t *stats = (ht_leaf_stats_t *)(pool->stats_storage + ((size_t)i * stride));
        stats->last_split_attempt_at = (ht_float_t)-1.0;
        stats->next_free = pool->free_stats_head;
        pool->free_stats_head = (uint16_t)i;
    }
    pool->stats_allocated = 0;

    return true;
}

void ht_memory_init(ht_memory_pool_t *pool) {
#ifdef HT_CLASSIF_ONLY
    (void)ht_memory_init_stride(pool, HT_MAX_FEATURES, HT_TASK_CLASSIFICATION);
#else
    (void)ht_memory_init_stride(pool, HT_MAX_FEATURES, HT_TASK_REGRESSION);
#endif
}

uint16_t ht_node_alloc(ht_memory_pool_t *pool) {
    if (!pool || pool->free_nodes_head == HT_INVALID_IDX) {
        return HT_INVALID_IDX;
    }

    uint16_t idx = pool->free_nodes_head;
    pool->free_nodes_head = pool->node_pool[idx].next_free;
    pool->nodes_allocated++;

    ht_node_t *node = &pool->node_pool[idx];
    memset(node, 0, sizeof(*node));
    node->stats_idx = HT_INVALID_IDX;
    node->flags = (uint8_t)(HT_NODE_IS_LEAF | HT_NODE_IN_USE);

    return idx;
}

void ht_node_free(ht_memory_pool_t *pool, uint16_t node_idx) {
    if (!pool || node_idx >= HT_MAX_NODES || node_idx == HT_INVALID_IDX) {
        return;
    }

    ht_node_t *node = &pool->node_pool[node_idx];
    if (!(node->flags & HT_NODE_IN_USE)) {
        return; /* Block double-free */
    }

    /* Reset node state and clear active flag */
    memset(node, 0, sizeof(*node));
    node->flags = 0;

    /* Push onto intrusive free-list */
    node->next_free = pool->free_nodes_head;
    pool->free_nodes_head = node_idx;

    if (pool->nodes_allocated > 0) {
        pool->nodes_allocated--;
    }
}

uint16_t ht_stats_alloc(ht_memory_pool_t *pool) {
    if (!pool || pool->free_stats_head == HT_INVALID_IDX) {
        return HT_INVALID_IDX;
    }

    uint16_t idx = pool->free_stats_head;
    ht_leaf_stats_t *stats = ht_stats_get(pool, idx);
    if (!stats) {
        return HT_INVALID_IDX;
    }

    pool->free_stats_head = stats->next_free;
    pool->stats_allocated++;

    memset(stats, 0, pool->leaf_stride);
    stats->last_split_attempt_at = (ht_float_t)0.0;

    return idx;
}

void ht_stats_free(ht_memory_pool_t *pool, uint16_t stats_idx) {
    if (!pool || stats_idx >= pool->max_leaves_capacity || stats_idx == HT_INVALID_IDX) {
        return;
    }

    ht_leaf_stats_t *stats = ht_stats_get(pool, stats_idx);
    if (!stats || stats->last_split_attempt_at < (ht_float_t)0.0) {
        return; /* Block double-free */
    }

    memset(stats, 0, pool->leaf_stride);
    stats->last_split_attempt_at = (ht_float_t)-1.0;
    stats->next_free = pool->free_stats_head;
    pool->free_stats_head = stats_idx;

    if (pool->stats_allocated > 0) {
        pool->stats_allocated--;
    }
}

size_t ht_memory_bytes_used(const ht_memory_pool_t *pool) {
    if (!pool) {
        return 0;
    }
    return ((size_t)pool->nodes_allocated * sizeof(ht_node_t)) +
           ((size_t)pool->stats_allocated * (size_t)pool->leaf_stride);
}

size_t ht_memory_total_capacity_bytes(void) {
    return sizeof(ht_memory_pool_t);
}

ht_node_t *ht_node_get(ht_memory_pool_t *pool, uint16_t idx) {
    if (!pool || idx >= HT_MAX_NODES || idx == HT_INVALID_IDX) {
        return NULL;
    }
    return &pool->node_pool[idx];
}

const ht_node_t *ht_node_get_const(const ht_memory_pool_t *pool, uint16_t idx) {
    if (!pool || idx >= HT_MAX_NODES || idx == HT_INVALID_IDX) {
        return NULL;
    }
    return &pool->node_pool[idx];
}

ht_leaf_stats_t *ht_stats_get(ht_memory_pool_t *pool, uint16_t idx) {
    if (!pool || idx >= pool->max_leaves_capacity || idx == HT_INVALID_IDX) {
        return NULL;
    }
    return (ht_leaf_stats_t *)(pool->stats_storage + ((size_t)idx * (size_t)pool->leaf_stride));
}

const ht_leaf_stats_t *ht_stats_get_const(const ht_memory_pool_t *pool, uint16_t idx) {
    if (!pool || idx >= pool->max_leaves_capacity || idx == HT_INVALID_IDX) {
        return NULL;
    }
    return (const ht_leaf_stats_t *)(pool->stats_storage + ((size_t)idx * (size_t)pool->leaf_stride));
}

uint16_t ht_node_available(const ht_memory_pool_t *pool) {
    if (!pool || pool->nodes_allocated >= HT_MAX_NODES) {
        return 0;
    }
    return (uint16_t)(HT_MAX_NODES - pool->nodes_allocated);
}

uint16_t ht_stats_available(const ht_memory_pool_t *pool) {
    if (!pool || pool->stats_allocated >= pool->max_leaves_capacity) {
        return 0;
    }
    return (uint16_t)(pool->max_leaves_capacity - pool->stats_allocated);
}

uint16_t ht_node_count(const ht_memory_pool_t *pool) {
    return pool ? pool->nodes_allocated : 0;
}

uint16_t ht_stats_count(const ht_memory_pool_t *pool) {
    return pool ? pool->stats_allocated : 0;
}
