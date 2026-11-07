/**
 * @file test_memory.c
 * @brief Level 1 (Unit) and Level 2 (Integration / Stress) Tests for the
 *        Deterministic Slab / Pool Allocator (ht_memory).
 */

#include "test_harness.h"
#include "ht_memory.h"

/* Static allocation to avoid blowing stack */
static ht_memory_pool_t s_test_pool;

/* --- Level 1: Unit Tests --- */

static void test_memory_init_state(void) {
    ht_memory_init(&s_test_pool);

    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), HT_MAX_NODES);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), HT_MAX_ACTIVE_LEAVES);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), 0);

    /* Total static capacity must strictly obey the <= 1 MB budget */
    HT_ASSERT(ht_memory_total_capacity_bytes() <= (1024 * 1024));
}

static void test_node_alloc_single_and_zeroed(void) {
    ht_memory_init(&s_test_pool);

    uint16_t n_idx = ht_node_alloc(&s_test_pool);
    HT_ASSERT_EQ(n_idx, 0); /* First allocation returns index 0 */
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 1);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), HT_MAX_NODES - 1);

    ht_node_t *node = ht_node_get(&s_test_pool, n_idx);
    HT_ASSERT_NOT_NULL(node);

    /* Verify default leaf state and initialization */
    HT_ASSERT((node->flags & HT_NODE_IS_LEAF) != 0);
    HT_ASSERT((node->flags & HT_NODE_IN_USE) != 0);
    HT_ASSERT_EQ(node->left_child, HT_INVALID_IDX);
    HT_ASSERT_EQ(node->right_child, HT_INVALID_IDX);
    HT_ASSERT_EQ(node->stats_idx, HT_INVALID_IDX);
    HT_ASSERT_EQ(node->feature_idx, HT_NO_FEATURE);

    for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
        HT_ASSERT_FLOAT_EQ(node->class_counts[c], (ht_float_t)0.0);
    }
    HT_ASSERT_FLOAT_EQ(node->split_val.threshold, (ht_float_t)0.0);

    /* Free node and verify recovery */
    ht_node_free(&s_test_pool, n_idx);
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), HT_MAX_NODES);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), 0);
}

static void test_stats_alloc_single_and_zeroed(void) {
    ht_memory_init(&s_test_pool);

    uint16_t s_idx = ht_stats_alloc(&s_test_pool);
    HT_ASSERT_EQ(s_idx, 0); /* First allocation returns index 0 */
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 1);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), HT_MAX_ACTIVE_LEAVES - 1);

    ht_leaf_stats_t *stats = ht_stats_get(&s_test_pool, s_idx);
    HT_ASSERT_NOT_NULL(stats);

    HT_ASSERT_FLOAT_EQ(stats->last_split_attempt_at, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(stats->mc_correct_weight, (ht_float_t)0.0);
    HT_ASSERT_FLOAT_EQ(stats->nb_correct_weight, (ht_float_t)0.0);

    /* Verify observers are zero-initialized */
    for (uint8_t f = 0; f < HT_MAX_FEATURES; ++f) {
        for (uint8_t c = 0; c < HT_MAX_CLASSES; ++c) {
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].count, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].mean, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].m2, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].min_val, (ht_float_t)0.0);
            HT_ASSERT_FLOAT_EQ(stats->observers[f].num[c].max_val, (ht_float_t)0.0);
        }
    }

    /* Free stats and verify recovery */
    ht_stats_free(&s_test_pool, s_idx);
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), HT_MAX_ACTIVE_LEAVES);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), 0);
}

static void test_node_pool_exhaustion_and_recovery(void) {
    ht_memory_init(&s_test_pool);
    static uint16_t allocated_indices[HT_MAX_NODES];

    /* Allocate all available nodes */
    for (uint32_t i = 0; i < HT_MAX_NODES; ++i) {
        uint16_t idx = ht_node_alloc(&s_test_pool);
        HT_ASSERT_NEQ(idx, HT_INVALID_IDX);
        allocated_indices[i] = idx;
    }

    HT_ASSERT_EQ(ht_node_count(&s_test_pool), HT_MAX_NODES);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), 0);

    /* Next allocation MUST fail with HT_INVALID_IDX */
    uint16_t overflow_idx = ht_node_alloc(&s_test_pool);
    HT_ASSERT_EQ(overflow_idx, HT_INVALID_IDX);

    /* Free all nodes in forward order */
    for (uint32_t i = 0; i < HT_MAX_NODES; ++i) {
        ht_node_free(&s_test_pool, allocated_indices[i]);
    }

    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), HT_MAX_NODES);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), 0);

    /* Reallocate all nodes again to confirm complete recyclability */
    for (uint32_t i = 0; i < HT_MAX_NODES; ++i) {
        uint16_t idx = ht_node_alloc(&s_test_pool);
        HT_ASSERT_NEQ(idx, HT_INVALID_IDX);
        allocated_indices[i] = idx;
    }
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), HT_MAX_NODES);

    /* Free all in reverse order */
    for (int32_t i = (int32_t)HT_MAX_NODES - 1; i >= 0; --i) {
        ht_node_free(&s_test_pool, allocated_indices[i]);
    }
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
}

static void test_stats_pool_exhaustion_and_recovery(void) {
    ht_memory_init(&s_test_pool);
    static uint16_t allocated_indices[HT_MAX_ACTIVE_LEAVES];

    /* Allocate all available stats blocks */
    for (uint32_t i = 0; i < HT_MAX_ACTIVE_LEAVES; ++i) {
        uint16_t idx = ht_stats_alloc(&s_test_pool);
        HT_ASSERT_NEQ(idx, HT_INVALID_IDX);
        allocated_indices[i] = idx;
    }

    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), HT_MAX_ACTIVE_LEAVES);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), 0);

    /* Next allocation MUST fail with HT_INVALID_IDX */
    uint16_t overflow_idx = ht_stats_alloc(&s_test_pool);
    HT_ASSERT_EQ(overflow_idx, HT_INVALID_IDX);

    /* Free all stats blocks */
    for (uint32_t i = 0; i < HT_MAX_ACTIVE_LEAVES; ++i) {
        ht_stats_free(&s_test_pool, allocated_indices[i]);
    }

    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), HT_MAX_ACTIVE_LEAVES);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), 0);

    /* Reallocate all */
    for (uint32_t i = 0; i < HT_MAX_ACTIVE_LEAVES; ++i) {
        uint16_t idx = ht_stats_alloc(&s_test_pool);
        HT_ASSERT_NEQ(idx, HT_INVALID_IDX);
        allocated_indices[i] = idx;
    }
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), HT_MAX_ACTIVE_LEAVES);

    /* Clean up */
    for (uint32_t i = 0; i < HT_MAX_ACTIVE_LEAVES; ++i) {
        ht_stats_free(&s_test_pool, allocated_indices[i]);
    }
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
}

static void test_memory_edge_cases_and_null(void) {
    /* NULL pointer safety */
    ht_memory_init(NULL);
    HT_ASSERT_EQ(ht_node_alloc(NULL), HT_INVALID_IDX);
    ht_node_free(NULL, 0);
    HT_ASSERT_EQ(ht_stats_alloc(NULL), HT_INVALID_IDX);
    ht_stats_free(NULL, 0);
    HT_ASSERT_EQ(ht_memory_bytes_used(NULL), 0);
    HT_ASSERT_NULL(ht_node_get(NULL, 0));
    HT_ASSERT_NULL(ht_node_get_const(NULL, 0));
    HT_ASSERT_NULL(ht_stats_get(NULL, 0));
    HT_ASSERT_NULL(ht_stats_get_const(NULL, 0));
    HT_ASSERT_EQ(ht_node_available(NULL), 0);
    HT_ASSERT_EQ(ht_stats_available(NULL), 0);
    HT_ASSERT_EQ(ht_node_count(NULL), 0);
    HT_ASSERT_EQ(ht_stats_count(NULL), 0);

    /* Out of bounds and invalid index queries on valid pool */
    ht_memory_init(&s_test_pool);
    HT_ASSERT_NULL(ht_node_get(&s_test_pool, HT_MAX_NODES));
    HT_ASSERT_NULL(ht_node_get(&s_test_pool, (uint16_t)(HT_MAX_NODES + 10)));
    HT_ASSERT_NULL(ht_node_get(&s_test_pool, HT_INVALID_IDX));

    HT_ASSERT_NULL(ht_stats_get(&s_test_pool, HT_MAX_ACTIVE_LEAVES));
    HT_ASSERT_NULL(ht_stats_get(&s_test_pool, (uint16_t)(HT_MAX_ACTIVE_LEAVES + 5)));
    HT_ASSERT_NULL(ht_stats_get(&s_test_pool, HT_INVALID_IDX));

    /* Freeing invalid or out of bounds indices must be no-ops */
    ht_node_free(&s_test_pool, HT_INVALID_IDX);
    ht_node_free(&s_test_pool, HT_MAX_NODES);
    ht_node_free(&s_test_pool, (uint16_t)(HT_MAX_NODES + 100));

    ht_stats_free(&s_test_pool, HT_INVALID_IDX);
    ht_stats_free(&s_test_pool, HT_MAX_ACTIVE_LEAVES);
    ht_stats_free(&s_test_pool, (uint16_t)(HT_MAX_ACTIVE_LEAVES + 50));

    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
}

static void test_memory_double_free_protection(void) {
    ht_memory_init(&s_test_pool);

    uint16_t n = ht_node_alloc(&s_test_pool);
    HT_ASSERT_EQ(n, 0);
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 1);

    ht_node_free(&s_test_pool, n);
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), HT_MAX_NODES);

    /* Second free of same index must be safely ignored */
    ht_node_free(&s_test_pool, n);
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), HT_MAX_NODES);

    uint16_t s = ht_stats_alloc(&s_test_pool);
    HT_ASSERT_EQ(s, 0);
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 1);

    ht_stats_free(&s_test_pool, s);
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), HT_MAX_ACTIVE_LEAVES);

    /* Second free of same stats must be safely ignored */
    ht_stats_free(&s_test_pool, s);
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), HT_MAX_ACTIVE_LEAVES);
}

static void test_memory_bytes_used_calculation(void) {
    ht_memory_init(&s_test_pool);

    /* Allocate 10 nodes and 3 stats blocks */
    uint16_t nodes[10];
    uint16_t stats[3];

    for (int i = 0; i < 10; ++i) {
        nodes[i] = ht_node_alloc(&s_test_pool);
    }
    for (int i = 0; i < 3; ++i) {
        stats[i] = ht_stats_alloc(&s_test_pool);
    }

    size_t expected_bytes = 10 * sizeof(ht_node_t) + 3 * sizeof(ht_leaf_stats_t);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), expected_bytes);

    /* Free 4 nodes and 1 stats */
    for (int i = 0; i < 4; ++i) {
        ht_node_free(&s_test_pool, nodes[i]);
    }
    ht_stats_free(&s_test_pool, stats[0]);

    expected_bytes = 6 * sizeof(ht_node_t) + 2 * sizeof(ht_leaf_stats_t);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), expected_bytes);

    /* Free remainder */
    for (int i = 4; i < 10; ++i) {
        ht_node_free(&s_test_pool, nodes[i]);
    }
    for (int i = 1; i < 3; ++i) {
        ht_stats_free(&s_test_pool, stats[i]);
    }

    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), 0);
}

/* --- Level 2: Meso / Integration & Stress Recycling Test --- */

static void test_memory_stress_recycling_100k(void) {
    ht_memory_init(&s_test_pool);

    static uint16_t active_nodes[HT_MAX_NODES];
    static uint16_t active_stats[HT_MAX_ACTIVE_LEAVES];
    uint16_t num_active_nodes = 0;
    uint16_t num_active_stats = 0;

    /* Simple deterministic LCG pseudo-random number generator */
    uint32_t rng_state = 0x12345678U;
    #define NEXT_RAND() (rng_state = rng_state * 1664525U + 1013904223U)

    const uint32_t NUM_CYCLES = 100000U;

    for (uint32_t cycle = 0; cycle < NUM_CYCLES; ++cycle) {
        uint32_t r = NEXT_RAND();
        bool do_node = ((r & 0xFFU) < 180U); /* ~70% node ops, ~30% stats ops */
        bool do_alloc = ((r & 0x100U) != 0);

        if (do_node) {
            if (do_alloc && num_active_nodes < HT_MAX_NODES) {
                uint16_t idx = ht_node_alloc(&s_test_pool);
                HT_ASSERT_NEQ(idx, HT_INVALID_IDX);
                active_nodes[num_active_nodes++] = idx;
            } else if (!do_alloc && num_active_nodes > 0) {
                uint32_t pick = NEXT_RAND() % num_active_nodes;
                uint16_t idx_to_free = active_nodes[pick];
                active_nodes[pick] = active_nodes[--num_active_nodes];
                ht_node_free(&s_test_pool, idx_to_free);
            }
        } else {
            if (do_alloc && num_active_stats < HT_MAX_ACTIVE_LEAVES) {
                uint16_t idx = ht_stats_alloc(&s_test_pool);
                HT_ASSERT_NEQ(idx, HT_INVALID_IDX);
                active_stats[num_active_stats++] = idx;
            } else if (!do_alloc && num_active_stats > 0) {
                uint32_t pick = NEXT_RAND() % num_active_stats;
                uint16_t idx_to_free = active_stats[pick];
                active_stats[pick] = active_stats[--num_active_stats];
                ht_stats_free(&s_test_pool, idx_to_free);
            }
        }

        /* Periodic verification of internal consistency */
        if ((cycle % 10000U) == 0U) {
            HT_ASSERT_EQ(ht_node_count(&s_test_pool), num_active_nodes);
            HT_ASSERT_EQ(ht_stats_count(&s_test_pool), num_active_stats);
            HT_ASSERT_EQ(ht_node_available(&s_test_pool), (uint16_t)(HT_MAX_NODES - num_active_nodes));
            HT_ASSERT_EQ(ht_stats_available(&s_test_pool), (uint16_t)(HT_MAX_ACTIVE_LEAVES - num_active_stats));
        }
    }

    #undef NEXT_RAND

    /* Clean up all remaining active allocations */
    for (uint16_t i = 0; i < num_active_nodes; ++i) {
        ht_node_free(&s_test_pool, active_nodes[i]);
    }
    for (uint16_t i = 0; i < num_active_stats; ++i) {
        ht_stats_free(&s_test_pool, active_stats[i]);
    }

    /* Final verification: zero leaks, 100% capacity restored */
    HT_ASSERT_EQ(ht_node_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_stats_count(&s_test_pool), 0);
    HT_ASSERT_EQ(ht_node_available(&s_test_pool), HT_MAX_NODES);
    HT_ASSERT_EQ(ht_stats_available(&s_test_pool), HT_MAX_ACTIVE_LEAVES);
    HT_ASSERT_EQ(ht_memory_bytes_used(&s_test_pool), 0);
}

void test_memory_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Slab / Pool Allocator (ht_memory)");
    HT_RUN_TEST(test_memory_init_state);
    HT_RUN_TEST(test_node_alloc_single_and_zeroed);
    HT_RUN_TEST(test_stats_alloc_single_and_zeroed);
    HT_RUN_TEST(test_node_pool_exhaustion_and_recovery);
    HT_RUN_TEST(test_stats_pool_exhaustion_and_recovery);
    HT_RUN_TEST(test_memory_edge_cases_and_null);
    HT_RUN_TEST(test_memory_double_free_protection);
    HT_RUN_TEST(test_memory_bytes_used_calculation);
    HT_RUN_TEST(test_memory_stress_recycling_100k);
}
