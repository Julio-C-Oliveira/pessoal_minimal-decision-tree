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
    params.delta = (ht_float_t)1e-2;
    params.leaf_prediction = HT_LEAF_MC;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 300; ++i) {
        uint8_t c = (i % 2 == 0) ? 0U : 1U;
        x[0].num = (c == 0U) ? (ht_float_t)(1.0 + (i % 7) * 0.1) : (ht_float_t)(10.0 + (i % 7) * 0.1);
        x[1].num = (ht_float_t)(i % 5);
        x[2].num = (ht_float_t)((i * 3) % 11);
        HT_ASSERT_EQ(ht_tree_learn_one(&tree, x, c, (ht_float_t)1.0), HT_OK);
    }
    HT_ASSERT(tree.n_decision_nodes >= 1U);

    ht_compact_tree_t ct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&ct, &tree), HT_OK);
    HT_ASSERT_EQ(ct.root_idx, 0U);
    HT_ASSERT_EQ(ct.num_nodes, ht_node_count(&tree.memory));

    /* Verify exact prediction parity (including NaN missing values) */
    for (int i = 0; i < 20; ++i) {
        x[0].num = (i < 10) ? (ht_float_t)(1.2 + i * 0.05) : (ht_float_t)(10.2 + (i - 10) * 0.05);
        x[1].num = (ht_float_t)i;
        x[2].num = (i == 5) ? ((ht_float_t)0.0 / (ht_float_t)0.0) : (ht_float_t)0.0;

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

    x[0].num = (ht_float_t)1.3;
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct_loaded, x), 0U);
    x[0].num = (ht_float_t)10.3;
    HT_ASSERT_EQ(ht_compact_tree_predict(&ct_loaded, x), 1U);
}

static void test_compact_tree_reg_and_subspace(void) {
    ht_config_params_t params;
    ht_default_reg_params(&params, 2);
    params.grace_period = 20U;
    params.delta = (ht_float_t)1e-2;
    params.leaf_prediction = HT_LEAF_MEAN;

    ht_tree_t tree;
    HT_ASSERT_EQ(ht_tree_init(&tree, &params), HT_OK);

    ht_value_t x[HT_MAX_FEATURES];
    memset(x, 0, sizeof(x));

    for (int i = 0; i < 200; ++i) {
        ht_float_t v = (i % 2 == 0) ? (ht_float_t)(1.0 + (i % 5) * 0.1) : (ht_float_t)(10.0 + (i % 5) * 0.1);
        ht_float_t y = (i % 2 == 0) ? (ht_float_t)5.0 : (ht_float_t)50.0;
        x[0].num = v;
        x[1].num = (ht_float_t)(i % 3);
        HT_ASSERT_EQ(ht_tree_learn_one_reg(&tree, x, y, (ht_float_t)1.0), HT_OK);
    }

    ht_compact_tree_t ct;
    HT_ASSERT_EQ(ht_compact_tree_from_tree(&ct, &tree), HT_OK);

    x[0].num = (ht_float_t)1.2;
    x[1].num = (ht_float_t)1.0;
    ht_float_t y_tree = ht_tree_predict_reg(&tree, x);
    ht_float_t y_ct = ht_compact_tree_predict_reg(&ct, x);
    HT_ASSERT_FLOAT_EQ(y_ct, y_tree);

    /* Subspace projection: map local feature 0 -> global feature 4, local 1 -> global 5 */
    uint8_t fmap[2] = {4U, 5U};
    HT_ASSERT_EQ(ht_compact_tree_set_subspace(&ct, fmap, 2U), HT_OK);
    ht_value_t x_global[HT_MAX_FEATURES];
    memset(x_global, 0, sizeof(x_global));
    x_global[4].num = (ht_float_t)10.2;
    x_global[5].num = (ht_float_t)1.0;
    ht_float_t y_sub = ht_compact_tree_predict_reg(&ct, x_global);
    HT_ASSERT(y_sub > (ht_float_t)30.0);
}

static void test_voting_weighted_and_uniform(void) {
    ht_float_t err_good = (ht_float_t)0.5;
    ht_float_t err_bad = (ht_float_t)0.5;
    for (int i = 0; i < 50; ++i) {
        ht_voting_update_error_classif(&err_good, 1U, 1U, (ht_float_t)0.9);
        ht_voting_update_error_classif(&err_bad, 0U, 1U, (ht_float_t)0.9);
    }
    HT_ASSERT(err_good < (ht_float_t)0.05);
    HT_ASSERT(err_bad > (ht_float_t)0.95);

    ht_float_t w[2];
    w[0] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_CLASSIFICATION, err_good);
    w[1] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_CLASSIFICATION, err_bad);
    HT_ASSERT(w[0] > w[1] * (ht_float_t)100.0);

    ht_float_t probas[2 * HT_MAX_CLASSES];
    memset(probas, 0, sizeof(probas));
    probas[0 * HT_MAX_CLASSES + 1] = (ht_float_t)0.9;
    probas[0 * HT_MAX_CLASSES + 0] = (ht_float_t)0.1;
    probas[1 * HT_MAX_CLASSES + 0] = (ht_float_t)0.9;
    probas[1 * HT_MAX_CLASSES + 1] = (ht_float_t)0.1;

    ht_float_t out[HT_MAX_CLASSES];
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, w, 2U, out), HT_OK);
    HT_ASSERT(out[1] > (ht_float_t)0.85);

    /* Uniform voting should yield 0.5 / 0.5 */
    ht_float_t wu[2] = {(ht_float_t)1.0, (ht_float_t)1.0};
    HT_ASSERT_EQ(ht_voting_combine_proba(2U, probas, wu, 2U, out), HT_OK);
    HT_ASSERT_FLOAT_NEAR(out[0], (ht_float_t)0.5, (ht_float_t)1e-5);
    HT_ASSERT_FLOAT_NEAR(out[1], (ht_float_t)0.5, (ht_float_t)1e-5);

    /* Regression weighted combination */
    ht_float_t mse_good = (ht_float_t)10.0, mse_bad = (ht_float_t)10.0;
    for (int i = 0; i < 40; ++i) {
        ht_voting_update_error_reg(&mse_good, (ht_float_t)5.0, (ht_float_t)5.0, (ht_float_t)0.9);
        ht_voting_update_error_reg(&mse_bad, (ht_float_t)0.0, (ht_float_t)10.0, (ht_float_t)0.9);
    }
    ht_float_t wr[2];
    wr[0] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, mse_good);
    wr[1] = ht_voting_member_weight(HT_VOTE_WEIGHTED, HT_TASK_REGRESSION, mse_bad);
    ht_float_t preds[2] = {(ht_float_t)5.0, (ht_float_t)50.0};
    ht_float_t combined = ht_voting_combine_reg(2U, preds, wr);
    HT_ASSERT(combined < (ht_float_t)7.0);
}

void test_compact_tree_run(void) {
    HT_TEST_SUITE_BEGIN("Nivel 1 & 2: Arvore Compacta e Motor de Votacao (ensemble/)");
    HT_RUN_TEST(test_compact_tree_classif_parity_and_serialization);
    HT_RUN_TEST(test_compact_tree_reg_and_subspace);
    HT_RUN_TEST(test_voting_weighted_and_uniform);
}
