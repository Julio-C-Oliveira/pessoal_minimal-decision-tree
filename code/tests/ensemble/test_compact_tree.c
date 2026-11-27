/**
 * @file test_compact_tree.c
 * @brief Level 1 & 2 tests for ht_compact_tree and ht_voting (shared ensemble foundation).
 */

#include "test_harness.h"
#include "ht_tree.h"
#include "ht_compact_tree.h"
#include "ht_voting.h"
#include <string.h>

static void test_compact_tree_classif_parity_and_serialization(void) {
    ht_config_params_t params;
    ht_default_params(&params, 3, 2);
    params.grace_period = 20U;
    params.delta = HT_CONST(1e-2);
    params.leaf_prediction = HT_LEAF_MC;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 300; ++i) {
        uint8_t c = (i % 2 == 0) ? 0U : 1U;
        ht_float_t jitter = HT_MUL(HT_FROM_INT(i % 7), HT_CONST(0.1));
        x[0].num = (c == 0U) ? HT_ADD(HT_CONST(1.0), jitter) : HT_ADD(HT_CONST(10.0), jitter);
        x[1].num = HT_FROM_INT(i % 5);
        x[2].num = HT_FROM_INT((i * 3) % 11);
        HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, c, HT_CONST(1.0)), HT_OK);
    }
    HT_ASSERT(tree.n_decision_nodes >= 1U);

    ht_compact_tree_t ct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&ct, &tree), HT_OK);
    HT_ASSERT_EQ(ct.root_idx, 0U);
    HT_ASSERT_EQ(ct.num_nodes, ht_node_count(&tree.memory));

    /* Verify exact prediction parity (including NaN missing values) */
    for (int i = 0; i < 20; ++i) {
        x[0].num = (i < 10)
            ? HT_ADD(HT_CONST(1.2), HT_MUL(HT_FROM_INT(i), HT_CONST(0.05)))
            : HT_ADD(HT_CONST(10.2), HT_MUL(HT_FROM_INT(i - 10), HT_CONST(0.05)));
        x[1].num = HT_FROM_INT(i);
        x[2].num = (i == 5) ? HT_NAN_VAL : HT_CONST(0.0);

        ht_float_t p_orig[HT_MAX_CLASSES], p_ct[HT_MAX_CLASSES];
        HT_ASSERT_EQ(ht_tree_predict_proba(&tree, x, p_orig), HT_OK);
        HT_ASSERT_EQ(ht_compact_tree_predict_proba(&ct, x, p_ct), HT_OK);
        for (uint8_t c = 0; c < 2; ++c) {
            HT_ASSERT_FLOAT_EQ(p_ct[c], p_orig[c]);
        }
        HT_ASSERT_EQ(ht_compact_tree_predict(&ct, x), ht_tree_predict(&tree, x));
    }

    /* Binary save/load roundtrip */
    size_t req_bytes = 0;
    HT_ASSERT_EQ(ht_compact_tree_save(&ct, NULL, 0, &req_bytes), HT_OK);
    HT_ASSERT(req_bytes > sizeof(ht_compact_serial_header_t));

    uint8_t buf[4096];
    HT_ASSERT(req_bytes <= sizeof(buf));
    size_t written = 0;
    HT_ASSERT_EQ(ht_compact_tree_save(&ct, buf, sizeof(buf), &written), HT_OK);
    HT_ASSERT_EQ(written, req_bytes);

    ht_compact_tree_t ct_loaded;
    HT_ASSERT_EQ(ht_compact_tree_load(&ct_loaded, buf, written), HT_OK);
    HT_ASSERT_EQ(ct_loaded.num_nodes, ct.num_nodes);
    HT_ASSERT_EQ(ct_loaded.n_features, ct.n_features);

    x[0].num = HT_CONST(1.3);
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct_loaded, x), 0U);
    x[0].num = HT_CONST(10.3);
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct_loaded, x), 1U);
}

#ifndef HT_CLASSIF_ONLY
static void test_compact_tree_reg_and_subspace(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.grace_period = 20U;
    params.delta = HT_CONST(1e-2);
    params.leaf_prediction = HT_LEAF_MEAN;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 200; ++i) {
        ht_float_t jitter = HT_MUL(HT_FROM_INT(i % 5), HT_CONST(0.1));
        ht_float_t v = (i % 2 == 0) ? HT_ADD(HT_CONST(1.0), jitter) : HT_ADD(HT_CONST(10.0), jitter);
        ht_float_t y = (i % 2 == 0) ? HT_CONST(5.0) : HT_CONST(50.0);
        x[0].num = v;
        x[1].num = HT_FROM_INT(i % 3);
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, y, HT_CONST(1.0)), HT_OK);
    }

    ht_compact_tree_t ct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&ct, &tree), HT_OK);

    x[0].num = HT_CONST(1.2);
    x[1].num = HT_CONST(1.0);
    ht_float_t y_tree = ht_tree_predict_reg(&tree, x);
    ht_float_t y_ct = ht_compact_tree_predict_reg(&ct, x);
    HT_ASSERT_FLOAT_EQ(y_ct, y_tree);

    /* Subspace projection: map local feature 0 -> global feature 4, local 1 -> global 5 */
    uint8_t fmap[2] = {4U, 5U};
    HT_ASSERT_EQ(ht_compact_tree_set_subspace(&ct, fmap, 2U), HT_OK);
    ht_value_t x_global[HT_MAX_FEATURES];
    memset(x_global, 0, sizeof(x_global));
    x_global[4].num = HT_CONST(10.2);
    x_global[5].num = HT_CONST(1.0);
    ht_float_t y_sub = ht_compact_tree_predict_reg(&ct, x_global);
    HT_ASSERT(y_sub > HT_CONST(30.0));
}
#endif

static void test_voting_weighted_and_uniform(void) {
    ht_float_t err_good = HT_CONST(0.5);
    ht_float_t err_bad = HT_CONST(0.5);
    for (int i = 0; i < 50; ++i) {
        ht_voting_update_error_classif(&err_good, 1U, 1U, HT_CONST(0.9));
        ht_voting_update_error_classif(&err_bad, 0U, 1U, HT_CONST(0.9));
    }
    HT_ASSERT(err_good < HT_CONST(0.05));
    HT_ASSERT(err_bad > HT_CONST(0.95));

    ht_float_t w[2];
    w[0] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_CLASSIFICATION, err_good);
    w[1] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_CLASSIFICATION, err_bad);
    HT_ASSERT(w[0] > HT_MUL(w[1], HT_CONST(100.0)));

    ht_float_t probas[2 * HT_MAX_CLASSES];
    memset(probas, 0, sizeof(probas));
    probas[0 * HT_MAX_CLASSES + 1] = HT_CONST(0.9);
    probas[0 * HT_MAX_CLASSES + 0] = HT_CONST(0.1);
    probas[1 * HT_MAX_CLASSES + 0] = HT_CONST(0.9);
    probas[1 * HT_MAX_CLASSES + 1] = HT_CONST(0.1);

    ht_float_t out[HT_MAX_CLASSES];
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, w, 2U, out), HT_OK);
    HT_ASSERT(out[1] > HT_CONST(0.85));

    /* Uniform voting should yield 0.5 / 0.5 */
    ht_float_t wu[2] = {HT_CONST(1.0), HT_CONST(1.0)};
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, wu, 2U, out), HT_OK);
    HT_ASSERT_FLOAT_NEAR(out[0], HT_CONST(0.5), HT_TEST_TOL(1e-5));
    HT_ASSERT_FLOAT_NEAR(out[1], HT_CONST(0.5), HT_TEST_TOL(1e-5));

    /* Regression weighted combination */
    ht_float_t mse_good = HT_CONST(10.0), mse_bad = HT_CONST(10.0);
    for (int i = 0; i < 40; ++i) {
        ht_voting_update_error_reg(&mse_good, HT_CONST(5.0), HT_CONST(5.0), HT_CONST(0.9));
        ht_voting_update_error_reg(&mse_bad, HT_CONST(0.0), HT_CONST(10.0), HT_CONST(0.9));
    }
    ht_float_t wr[2];
    wr[0] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, mse_good);
    wr[1] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, mse_bad);
    ht_float_t preds[2] = {HT_CONST(5.0), HT_CONST(50.0)};
    ht_float_t combined = ht_voting_combine_reg(2U, preds, wr);
    HT_ASSERT(combined < HT_CONST(7.0));
}

static void test_compact_tree_nan_on_split_and_categorical(void) {
    /* Classification tree with numeric split (f0) and categorical split (f1) */
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    params.feature_types[0] = HT_FEAT_NUMERIC;
    params.feature_types[1] = HT_FEAT_CATEGORICAL;
    params.grace_period = 15U;
    params.delta = HT_CONST(1e-2);
    params.leaf_prediction = HT_LEAF_MC;

    static ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    /* Build an explicit tree with a numeric root (f0) and categorical right child (f1) */
    ht_split_candidate_t c_num;
    memset(&c_num, 0, sizeof(c_num));
    c_num.feature_idx = 0U;
    c_num.is_categorical = false;
    c_num.split_val.threshold = HT_CONST(5.0);
    c_num.merit = HT_CONST(0.9);
    uint16_t lc = 0, rc = 0;
    HT_ASSERT_EQ(ht_tree_apply_split(&tree, tree.root_idx, &c_num, 0U, &lc, &rc), HT_OK);

    ht_split_candidate_t c_cat;
    memset(&c_cat, 0, sizeof(c_cat));
    c_cat.feature_idx = 1U;
    c_cat.is_categorical = true;
    c_cat.split_val.cat_val = 2;
    c_cat.merit = HT_CONST(0.8);
    uint16_t r_lc = 0, r_rc = 0;
    HT_ASSERT_EQ(ht_tree_apply_split(&tree, rc, &c_cat, 1U, &r_lc, &r_rc), HT_OK);

    /* Populate leaf weights asymmetrically so most_common_child picks right (weight 40 vs 10) */
    tree.memory.node_pool[lc].class_counts[0] = HT_CONST(10.0);
    tree.memory.node_pool[r_lc].class_counts[1] = HT_CONST(30.0);
    tree.memory.node_pool[r_rc].class_counts[0] = HT_CONST(10.0);

    ht_compact_tree_t ct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&ct, &tree), HT_OK);

    /* Route with NaN on numeric split feature 0 AND invalid category on feature 1 */
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    x[0].num = HT_NAN_VAL;
    x[1].cat = -1; /* missing category */
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct, x), 1U);

    /* Valid categorical match vs non-match */
    x[0].num = HT_CONST(8.0);
    x[1].cat = 2;
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct, x), 1U);
    x[1].cat = 0;
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct, x), 0U);
    x[1].cat = HT_MAX_CATEGORIES; /* out-of-range missing category -> most common child (r_lc -> class 1) */
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct, x), 1U);

    /* Subspace projection in classification mode (works under HT_CLASSIF_ONLY too) */
    uint8_t fmap[2] = {3U, 4U};
    HT_ASSERT_EQ(ht_compact_tree_set_subspace(&ct, fmap, 2U), HT_OK);
    ht_value_t xg[HT_MAX_FEATURES];
    memset(xg, 0, sizeof(xg));
    xg[3].num = HT_CONST(8.0);
    xg[4].cat = 2;
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct, xg), 1U);

#ifndef HT_CLASSIF_ONLY
    /* Regression NaN on split feature -> exercises regression path of ht_compact_subtree_weight */
    ht_config_params_t rparams;
    ht_default_reg_params(&rparams, 2);
    static ht_tree_t rtree;
    HT_ASSERT_EQ(ht_tree_init(&rtree, &rparams), HT_OK);
    HT_ASSERT_EQ(ht_tree_apply_split(&rtree, rtree.root_idx, &c_num, 0U, &lc, &rc), HT_OK);
    rtree.memory.node_pool[lc].reg_stats.count = HT_CONST(5.0);
    rtree.memory.node_pool[lc].reg_stats.mean = HT_CONST(10.0);
    rtree.memory.node_pool[rc].reg_stats.count = HT_CONST(25.0);
    rtree.memory.node_pool[rc].reg_stats.mean = HT_CONST(90.0);

    ht_compact_tree_t rct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&rct, &rtree), HT_OK);
    x[0].num = HT_NAN_VAL;
    HT_ASSERT_FLOAT_NEAR(ht_compact_tree_predict_reg(&rct, x), HT_CONST(90.0), HT_TEST_TOL(1e-3));
#endif
}

static void test_compact_tree_capacity_truncation_bfs(void) {
#if (HT_MAX_NODES >= (HT_COMPACT_MAX_NODES + 16U))
    /* Build a full ht_tree_t exceeding HT_COMPACT_MAX_NODES so BFS triggers
     * graceful frontier truncation via ht_summarize_tree_subtree(). */
    ht_config_params_t params;
    ht_default_params(&params, 2, 2);
    static ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    uint16_t queue[HT_COMPACT_MAX_NODES + 32U];
    uint16_t qh = 0U, qt = 0U;
    queue[qt++] = tree.root_idx;

    ht_split_candidate_t cand;
    memset(&cand, 0, sizeof(cand));
    cand.feature_idx = 0U;
    cand.is_categorical = false;
    cand.merit = HT_CONST(0.5);

    uint16_t target_splits = (uint16_t)(HT_COMPACT_MAX_NODES / 2U + 4U);
    for (uint16_t s = 0; s < target_splits && qh < qt; ++s) {
        uint16_t leaf = queue[qh++];
        cand.split_val.threshold = HT_FROM_UINT(s + 1U);
        uint16_t lc = HT_INVALID_IDX, rc = HT_INVALID_IDX;
        if (ht_tree_apply_split(&tree, leaf, &cand, 1U, &lc, &rc) != HT_OK) {
            break;
        }
        tree.memory.node_pool[lc].class_counts[0] = HT_CONST(3.0);
        tree.memory.node_pool[rc].class_counts[1] = HT_CONST(7.0);
        if (qt + 2U < (uint16_t)(sizeof(queue) / sizeof(queue[0]))) {
            queue[qt++] = lc;
            queue[qt++] = rc;
        }
    }
    HT_ASSERT(ht_node_count(&tree.memory) > HT_COMPACT_MAX_NODES);

    ht_compact_tree_t ct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&ct, &tree), HT_OK);
    HT_ASSERT(ct.num_nodes <= HT_COMPACT_MAX_NODES);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    x[0].num = HT_CONST(500.0);
    ht_float_t probas[HT_MAX_CLASSES];
    HT_ASSERT_EQ(ht_compact_tree_predict_proba(&ct, x, probas), HT_OK);
    HT_ASSERT_FLOAT_NEAR(HT_ADD(probas[0], probas[1]), HT_CONST(1.0), HT_TEST_TOL(1e-3));

#ifndef HT_CLASSIF_ONLY
    /* Same BFS frontier truncation in regression mode */
    ht_config_params_t rparams;
    ht_default_reg_params(&rparams, 2);
    static ht_tree_t rtree;
    HT_ASSERT_EQ(ht_tree_init(&rtree, &rparams), HT_OK);
    qh = 0U; qt = 0U;
    queue[qt++] = rtree.root_idx;
    for (uint16_t s = 0; s < target_splits && qh < qt; ++s) {
        uint16_t leaf = queue[qh++];
        cand.split_val.threshold = HT_FROM_UINT(s + 1U);
        uint16_t lc = HT_INVALID_IDX, rc = HT_INVALID_IDX;
        if (ht_tree_apply_split(&rtree, leaf, &cand, 1U, &lc, &rc) != HT_OK) {
            break;
        }
        rtree.memory.node_pool[lc].reg_stats.count = HT_CONST(4.0);
        rtree.memory.node_pool[lc].reg_stats.mean = HT_CONST(20.0);
        rtree.memory.node_pool[rc].reg_stats.count = HT_CONST(4.0);
        rtree.memory.node_pool[rc].reg_stats.mean = HT_CONST(80.0);
        if (qt + 2U < (uint16_t)(sizeof(queue) / sizeof(queue[0]))) {
            queue[qt++] = lc;
            queue[qt++] = rc;
        }
    }
    ht_compact_tree_t rct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&rct, &rtree), HT_OK);
    HT_ASSERT(rct.num_nodes <= HT_COMPACT_MAX_NODES);
    ht_float_t rpred = ht_compact_tree_predict_reg(&rct, x);
    HT_ASSERT(HT_ISFINITE(rpred) && rpred > HT_CONST(0.0));
#endif
#endif
}

static void test_compact_tree_error_and_corrupt_serialization(void) {
    ht_compact_tree_t ct;
    memset(&ct, 0, sizeof(ct));
    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));
    ht_float_t probas[HT_MAX_CLASSES];

    /* NULL / uninitialized guards */
    HT_ASSERT_EQ(ht_compact_tree_from_tree(NULL, NULL), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_compact_tree_set_subspace(NULL, NULL, 0U), HT_ERR_INVALID_ARG);
    uint8_t fmap[2] = {0U, 1U};
    HT_ASSERT_EQ(ht_compact_tree_set_subspace(&ct, fmap, 0U), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_compact_tree_set_subspace(&ct, fmap, (uint8_t)(HT_MAX_FEATURES + 1U)), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_compact_tree_route(NULL, x), HT_INVALID_IDX);
    HT_ASSERT_EQ(ht_compact_tree_route(&ct, x), HT_INVALID_IDX);
    HT_ASSERT_EQ(ht_compact_tree_predict_proba(&ct, x, probas), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct, x), 0U);
    HT_ASSERT_FLOAT_EQ(ht_compact_tree_predict_reg(NULL, x), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_compact_tree_predict_reg(&ct, x), HT_CONST(0.0));

    /* Zero-weight leaf yields uniform probabilities */
    ct.num_nodes = 1U;
    ct.root_idx = 0U;
    ct.n_features = 2U;
    ct.n_classes = 2U;
    ct.nodes[0].flags = HT_NODE_IS_LEAF | HT_NODE_IN_USE;
    HT_ASSERT_EQ(ht_compact_tree_predict_proba(&ct, x, probas), HT_OK);
    HT_ASSERT_FLOAT_NEAR(probas[0], HT_CONST(0.5), HT_TEST_TOL(1e-4));
    HT_ASSERT_FLOAT_NEAR(probas[1], HT_CONST(0.5), HT_TEST_TOL(1e-4));

    /* Save / load error & corruption paths */
    uint8_t buf[512];
    size_t written = 0;
    HT_ASSERT_EQ(ht_compact_tree_save(NULL, buf, sizeof(buf), &written), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_compact_tree_save(&ct, buf, 4U, &written), HT_ERR_BUFFER_TOO_SMALL);
    HT_ASSERT_EQ(ht_compact_tree_save(&ct, buf, sizeof(buf), &written), HT_OK);

    ht_compact_tree_t loaded;
    HT_ASSERT_EQ(ht_compact_tree_load(NULL, buf, written), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_compact_tree_load(&loaded, NULL, written), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_compact_tree_load(&loaded, buf, 8U), HT_ERR_CORRUPT_IMAGE);

    /* Corrupt magic */
    uint8_t bad[512];
    memcpy(bad, buf, written);
    bad[0] ^= 0xFFU;
    HT_ASSERT_EQ(ht_compact_tree_load(&loaded, bad, written), HT_ERR_CORRUPT_IMAGE);

    /* Cross-precision flag mismatch (HT_COMPACT_FLAG_FIXED / DOUBLE) */
    ht_compact_serial_header_t hdr;
    memcpy(bad, buf, written);
    memcpy(&hdr, bad, sizeof(hdr));
    hdr.flags ^= HT_COMPACT_FLAG_FIXED;
    memcpy(bad, &hdr, sizeof(hdr));
    HT_ASSERT_EQ(ht_compact_tree_load(&loaded, bad, written), HT_ERR_CORRUPT_IMAGE);

    /* Truncated payload */
    HT_ASSERT_EQ(ht_compact_tree_load(&loaded, buf, sizeof(ht_compact_serial_header_t)), HT_ERR_CORRUPT_IMAGE);

    /* Out-of-bounds child index in loaded branch node */
    ct.num_nodes = 1U;
    ct.nodes[0].flags = HT_NODE_IN_USE; /* internal node with invalid children */
    ct.nodes[0].left_child = 99U;
    ct.nodes[0].right_child = 100U;
    HT_ASSERT_EQ(ht_compact_tree_save(&ct, buf, sizeof(buf), &written), HT_OK);
    HT_ASSERT_EQ(ht_compact_tree_load(&loaded, buf, written), HT_ERR_CORRUPT_IMAGE);

    /* Route guard on out-of-range feature_idx / subspace global index */
    ct.nodes[0].feature_idx = HT_MAX_FEATURES;
    HT_ASSERT_EQ(ht_compact_tree_route(&ct, x), HT_INVALID_IDX);
    ct.nodes[0].feature_idx = 0U;
    ct.has_subspace = 1U;
    ct.feature_map[0] = HT_MAX_FEATURES;
    HT_ASSERT_EQ(ht_compact_tree_route(&ct, x), HT_INVALID_IDX);
}

static void test_voting_extreme_mse_saturation_and_edges(void) {
    /* Reproduces Phase 8 Fed-Forest high-amplitude regression MSE saturation bug:
     * errors of 500..10000 square past Q16.16 max (32767.99), which previously
     * saturated faded_mse to +HT_INFINITY -> weight = 0 -> combined pred = 0.0. */
    ht_float_t mse_a = HT_CONST(10.0);
    ht_float_t mse_b = HT_CONST(10.0);
    for (int i = 0; i < 30; ++i) {
        ht_voting_update_error_reg(&mse_a, HT_CONST(1000.0), HT_CONST(1200.0), HT_CONST(0.95));
        ht_voting_update_error_reg(&mse_b, HT_CONST(1000.0), HT_CONST(5000.0), HT_CONST(0.95));
    }
    HT_ASSERT(HT_ISFINITE(mse_a));
    HT_ASSERT(HT_ISFINITE(mse_b));

    ht_float_t wa = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, mse_a);
    ht_float_t wb = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, mse_b);
    HT_ASSERT(wa > HT_CONST(0.0));
    HT_ASSERT(wb > HT_CONST(0.0));

    ht_float_t preds[2] = { HT_CONST(1150.0), HT_CONST(4800.0) };
    ht_float_t ws[2] = { wa, wb };
    ht_float_t y_comb = ht_voting_combine_reg(2U, preds, ws);
    HT_ASSERT(HT_ISFINITE(y_comb));
    HT_ASSERT(y_comb >= HT_CONST(1100.0) && y_comb <= HT_CONST(4850.0));

    /* Uniform strategy & edge cases (NULL, n_models=0, all-zero weights) */
    HT_ASSERT_FLOAT_EQ(ht_voting_member_weight(HT_VOTE_UNIFORM, HT_TASK_CLASSIFICATION, HT_CONST(0.9)), HT_CONST(1.0));
    ht_voting_update_error_classif(NULL, 0U, 1U, HT_CONST(0.9));
    ht_voting_update_error_reg(NULL, HT_CONST(1.0), HT_CONST(2.0), HT_CONST(0.9));

    ht_float_t out[HT_MAX_CLASSES];
    ht_float_t probas[2 * HT_MAX_CLASSES];
    memset(probas, 0, sizeof(probas));
    probas[0] = HT_CONST(0.8); probas[1] = HT_CONST(0.2);
    probas[HT_MAX_CLASSES + 0] = HT_CONST(0.2); probas[HT_MAX_CLASSES + 1] = HT_CONST(0.8);
    ht_float_t zero_w[2] = { HT_CONST(0.0), HT_CONST(0.0) };
    HT_ASSERT_EQ(ht_voting_combine_proba(0U, probas, zero_w, 2U, out), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, zero_w, 2U, out), HT_OK);
    HT_ASSERT_FLOAT_NEAR(out[0], HT_CONST(0.5), HT_TEST_TOL(1e-4));

    HT_ASSERT_FLOAT_EQ(ht_voting_combine_reg(0U, preds, zero_w), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_voting_combine_reg(2U, preds, zero_w), HT_CONST(0.0));

    /* Negative / NaN / clamped decay and negative error edge branches in ht_voting */
    ht_float_t err_clamp = HT_CONST(0.5);
    ht_voting_update_error_classif(&err_clamp, 0U, 1U, HT_CONST(-0.5));
    ht_voting_update_error_classif(&err_clamp, 0U, 0U, HT_CONST(1.5));
    ht_float_t mse_clamp = HT_CONST(50000.0);
    ht_voting_update_error_reg(&mse_clamp, HT_CONST(1.0), HT_CONST(2.0), HT_CONST(-0.5));
    ht_voting_update_error_reg(&mse_clamp, HT_CONST(1.0), HT_CONST(2.0), HT_CONST(1.5));
    ht_voting_update_error_reg(&mse_clamp, HT_NAN_VAL, HT_CONST(2.0), HT_CONST(0.9));
    HT_ASSERT(ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_CLASSIFICATION, HT_CONST(-1.0)) > HT_CONST(0.0));
    HT_ASSERT(ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, HT_CONST(-1.0)) > HT_CONST(0.0));
    ht_float_t w_inf = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, HT_INFINITY);
    HT_ASSERT(HT_ISFINITE(w_inf) && w_inf >= HT_CONST(0.0));

    ht_float_t neg_w[2] = { HT_CONST(-1.0), HT_CONST(2.0) };
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, NULL, neg_w, 2U, out), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, NULL, 2U, out), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, neg_w, 0U, out), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, neg_w, 2U, NULL), HT_ERR_INVALID_ARG);
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, neg_w, 2U, out), HT_OK);
    HT_ASSERT_FLOAT_NEAR(ht_voting_combine_reg(2U, preds, neg_w), preds[1], HT_TEST_TOL(1e-2));
    HT_ASSERT_FLOAT_EQ(ht_voting_combine_reg(2U, NULL, neg_w), HT_CONST(0.0));
    HT_ASSERT_FLOAT_EQ(ht_voting_combine_reg(2U, preds, NULL), HT_CONST(0.0));
}

void test_compact_tree_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Arvore Compacta e Motor de Votacao (ensemble/)");
    HT_RUN_TEST(test_compact_tree_classif_parity_and_serialization);
#ifndef HT_CLASSIF_ONLY
    HT_RUN_TEST(test_compact_tree_reg_and_subspace);
#endif
    HT_RUN_TEST(test_voting_weighted_and_uniform);
    HT_RUN_TEST(test_compact_tree_nan_on_split_and_categorical);
    HT_RUN_TEST(test_compact_tree_capacity_truncation_bfs);
    HT_RUN_TEST(test_compact_tree_error_and_corrupt_serialization);
    HT_RUN_TEST(test_voting_extreme_mse_saturation_and_edges);
}
