/**
 * @file ht_memory.c
 * @brief Implementation of the Deterministic Slab / Pool Allocator for Hoeffding Tree.
 */

#include "ht_memory.h"

void ht_memory_init(ht_memory_pool_t *pool) {
    if (!pool) {
        return;
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

    /* Initialize stats intrusive free-list in descending order so index 0 is at head */
    pool->free_stats_head = HT_INVALID_IDX;
    for (int32_t i = (int32_t)HT_MAX_ACTIVE_LEAVES - 1; i >= 0; --i) {
        pool->stats_pool[i].last_split_attempt_at = (ht_float_t)-1.0;
        pool->stats_pool[i].next_free = pool->free_stats_head;
        pool->free_stats_head = (uint16_t)i;
    }
    pool->stats_allocated = 0;
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
    node->left_child = HT_INVALID_IDX;
    node->right_child = HT_INVALID_IDX;
    node->stats_idx = HT_INVALID_IDX;
    node->feature_idx = HT_NO_FEATURE;
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
    node->left_child = HT_INVALID_IDX;
    node->right_child = HT_INVALID_IDX;
    node->stats_idx = HT_INVALID_IDX;
    node->feature_idx = HT_NO_FEATURE;

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
    pool->free_stats_head = pool->stats_pool[idx].next_free;
    pool->stats_allocated++;

    ht_leaf_stats_t *stats = &pool->stats_pool[idx];
    memset(stats, 0, sizeof(*stats));
    stats->last_split_attempt_at = (ht_float_t)0.0;

    return idx;
}

void ht_stats_free(ht_memory_pool_t *pool, uint16_t stats_idx) {
    if (!pool || stats_idx >= HT_MAX_ACTIVE_LEAVES || stats_idx == HT_INVALID_IDX) {
        return;
    }

    ht_leaf_stats_t *stats = &pool->stats_pool[stats_idx];
    if (stats->last_split_attempt_at < (ht_float_t)0.0) {
        return; /* Block double-free */
    }

    memset(stats, 0, sizeof(*stats));
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
           ((size_t)pool->stats_allocated * sizeof(ht_leaf_stats_t));
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
    if (!pool || idx >= HT_MAX_ACTIVE_LEAVES || idx == HT_INVALID_IDX) {
        return NULL;
    }
    return &pool->stats_pool[idx];
}

const ht_leaf_stats_t *ht_stats_get_const(const ht_memory_pool_t *pool, uint16_t idx) {
    if (!pool || idx >= HT_MAX_ACTIVE_LEAVES || idx == HT_INVALID_IDX) {
        return NULL;
    }
    return &pool->stats_pool[idx];
}

uint16_t ht_node_available(const ht_memory_pool_t *pool) {
    if (!pool || pool->nodes_allocated >= HT_MAX_NODES) {
        return 0;
    }
    return (uint16_t)(HT_MAX_NODES - pool->nodes_allocated);
}

uint16_t ht_stats_available(const ht_memory_pool_t *pool) {
    if (!pool || pool->stats_allocated >= HT_MAX_ACTIVE_LEAVES) {
        return 0;
    }
    return (uint16_t)(HT_MAX_ACTIVE_LEAVES - pool->stats_allocated);
}

uint16_t ht_node_count(const ht_memory_pool_t *pool) {
    return pool ? pool->nodes_allocated : 0;
}

uint16_t ht_stats_count(const ht_memory_pool_t *pool) {
    return pool ? pool->stats_allocated : 0;
}
