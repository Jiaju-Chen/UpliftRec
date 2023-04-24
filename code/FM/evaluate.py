import numpy as np
import torch
import math
import time
from scipy.stats import entropy


def RMSE(model, model_name, dataloader):
    RMSE = np.array([], dtype=np.float32)
    for features, feature_values, label in dataloader:
        features = features.cuda()
        feature_values = feature_values.cuda()
        label = label.cuda()

        prediction = model(features, feature_values)
        prediction = prediction.clamp(min=-1.0, max=1.0)
        SE = (prediction - label).pow(2)
        RMSE = np.append(RMSE, SE.detach().cpu().numpy())

    return np.sqrt(RMSE.mean())


def pre_ranking(item_feature):
    '''prepare for the ranking: construct item_feature data'''

    features = []
    feature_values = []
    for itemID in range(len(item_feature)):
        features.append(np.array(item_feature[itemID][0]))
        feature_values.append(np.array(item_feature[itemID][1], dtype=np.float32))

    features = torch.tensor(features).cuda()
    feature_values = torch.tensor(feature_values).cuda()

    return features, feature_values


def selected_concat(user_feature, all_item_features, all_item_feature_values, userID, batch_item_idx):
    item_num = len(batch_item_idx)
    user_feat = torch.tensor(user_feature[userID][0]).cuda()
    user_feat = user_feat.expand(item_num, -1)
    user_feat_values = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32)).cuda()
    user_feat_values = user_feat_values.expand(item_num, -1)

    batch_item_idx = torch.tensor(batch_item_idx).cuda()
    batch_item_features = all_item_features[batch_item_idx]
    batch_item_feature_values = all_item_feature_values[batch_item_idx]

    features = torch.cat([user_feat, batch_item_features], 1)
    feature_values = torch.cat([user_feat_values, batch_item_feature_values], 1)

    return features, feature_values


def Ranking(model, valid_dict, test_dict, train_dict, user_feature, all_item_features, all_item_feature_values, \
            item_feature_dict, item_rank_dict, batch_size, topN, user_fml_cat, category_num, item_cat_vec,
            return_pred=False):
    """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
    user_gt_test = []
    user_gt_valid = []
    user_pred_for_valid = []
    user_pred_for_test = []
    indices_top1k = {}
    scores_top1k = {}
    fml_cat_list = []

    for userID in test_dict:
        #         features, feature_values, mask = user_rank_feature[userID]
        fml_cat_list.append(user_fml_cat[userID])
        batch_num = all_item_features.size(0) // batch_size
        item_idx = list(range(all_item_features.size(0)))
        st, ed = 0, batch_size
        mask_for_valid = torch.zeros(all_item_features.size(0)).cuda()
        mask_for_test = torch.zeros(all_item_features.size(0)).cuda()
        train_items = torch.tensor(train_dict[userID]).cuda()
        valid_items = torch.tensor(valid_dict[userID]).cuda()
        mask_for_valid[train_items] = -999
        mask_for_test[train_items] = -999
        mask_for_test[valid_items] = -999

        for i in range(batch_num):
            batch_item_idx = item_idx[st: ed]
            batch_feature, batch_feature_values = selected_concat(user_feature, all_item_features, \
                                                                  all_item_feature_values, userID, batch_item_idx)

            prediction = model(batch_feature, batch_feature_values)
            prediction = prediction
            if i == 0:
                all_predictions = prediction
            else:
                all_predictions = torch.cat([all_predictions, prediction], 0)

            st, ed = st + batch_size, ed + batch_size

        #         prediction for the last batch
        batch_item_idx = item_idx[st:]
        batch_feature, batch_feature_values = selected_concat(user_feature, all_item_features, \
                                                              all_item_feature_values, userID, batch_item_idx)

        prediction = model(batch_feature, batch_feature_values)
        if batch_num == 0:
            all_predictions = prediction
        else:
            all_predictions = torch.cat([all_predictions, prediction], 0)
        all_predictions_for_valid = all_predictions + mask_for_valid
        all_predictions_for_test = all_predictions + mask_for_test

        user_gt_valid.append(valid_dict[userID])
        user_gt_test.append(test_dict[userID])

        _, indices_for_valid = torch.topk(all_predictions_for_valid, topN[-1])
        _, indices_for_test = torch.topk(all_predictions_for_test, topN[-1])
        # inference
        scores, indices = torch.topk(all_predictions_for_test, min(1000, len(item_feature_dict)))
        indices_for_test = indices_for_test.detach().cpu().numpy()
        scores = sigmoid(scores.detach())

        pred_items_for_valid = torch.tensor(item_idx)[indices_for_valid].cpu().numpy().tolist()
        pred_items_for_test = torch.tensor(item_idx)[indices_for_test].cpu().numpy().tolist()

        indices_top1k[userID] = indices.detach()
        scores_top1k[userID] = scores
        user_pred_for_valid.append(pred_items_for_valid)
        user_pred_for_test.append(pred_items_for_test)

    valid_results = computeTopNAccuracy(user_gt_valid, user_pred_for_valid, topN, item_feature_dict, item_rank_dict)
    test_results = computeTopNAccuracy(user_gt_test, user_pred_for_test, topN, item_feature_dict, item_rank_dict)

    valid_results_unexp = computeUnexp(user_gt_valid, user_pred_for_valid, topN, item_feature_dict, item_rank_dict,
                                       fml_cat_list)
    test_results_unexp = computeUnexp(user_gt_test, user_pred_for_test, topN, item_feature_dict, item_rank_dict,
                                      fml_cat_list)

    valid_ent_gini = computeEntGini(user_gt_valid, user_pred_for_valid, topN, item_feature_dict, item_rank_dict,
                                    category_num)
    test_ent_gini = computeEntGini(user_gt_test, user_pred_for_test, topN, item_feature_dict, item_rank_dict,
                                   category_num)

    if return_pred:  # used in the inference.py
        return valid_results, test_results, valid_results_unexp, test_results_unexp, valid_ent_gini, test_ent_gini, indices_top1k, scores_top1k
    return valid_results, test_results, valid_results_unexp, test_results_unexp, valid_ent_gini, test_ent_gini


def sigmoid(x):
    s = 1 / (1 + torch.exp(-x))
    return s


def computeTopNAccuracy(GroundTruth, predictedIndices, topN, item_feature_dict, item_rank_dict):
    pop_rate = 0.1
    recall = []
    NDCG = []
    Cov_div = []
    Cov_nov = []
    Cov_pos = []
    recall_unpop = []

    for index in range(len(topN)):
        n_user = len(GroundTruth)
        n_user_with_unpop = len(GroundTruth)
        sumForRecall = 0
        sumForNdcg = 0
        sumForCov_div = 0
        sumForCov_nov = 0
        sumForCov_pos = 0
        sumForRecall_unpop = 0
        for i in range(len(predictedIndices)):  # for a user,
            len_test = len(GroundTruth[i])
            if len_test != 0:
                userHit = 0
                userHit_unpop = 0
                dcg = 0
                idcg = 0
                idcgCount = len_test
                ndcg = 0
                category_dict = {}
                category_pos_dict = {}
                novel_category_dict = {}
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        userHit += 1
                        if item_rank_dict[predictedIndices[i][j]] > math.floor(len(item_rank_dict) * pop_rate):
                            userHit_unpop += 1
                        for element in item_feature_dict[predictedIndices[i][j]]:
                            category_pos_dict[element] = 1
                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1
                    for element in item_feature_dict[predictedIndices[i][j]]:
                        # print(element)
                        category_dict[element] = 1

                        if item_rank_dict[predictedIndices[i][j]] > len(item_rank_dict) / 10:
                            novel_category_dict[element] = 1
                len_unpop = 0
                for item in GroundTruth[i]:
                    if item_rank_dict[item] > math.floor(len(item_rank_dict) * pop_rate):
                        len_unpop += 1

                if len_unpop == 0:
                    n_user_with_unpop -= 1
                else:
                    sumForRecall_unpop += userHit_unpop / len_unpop
                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForRecall += userHit / len_test
                sumForNdcg += ndcg
                sumForCov_div += len(category_dict)
                sumForCov_nov += len(novel_category_dict)
                sumForCov_pos += len(category_pos_dict)
            else:
                n_user -= 1
                n_user_with_unpop -= 1
        # print(n_user,n_user_with_unpop)
        recall.append(round(sumForRecall / n_user, 4))
        NDCG.append(round(sumForNdcg / n_user, 4))
        Cov_div.append(round(sumForCov_div / n_user, 4))
        Cov_nov.append(round(sumForCov_nov / n_user, 4))
        Cov_pos.append(round(sumForCov_pos / n_user, 4))
        if n_user_with_unpop == 0:
            recall_unpop.append(-999)
        else:
            recall_unpop.append(round(sumForRecall / n_user_with_unpop, 4))

    return recall, NDCG, Cov_div, Cov_nov, Cov_pos, recall_unpop


def computeUnexp(GroundTruth, predictedIndices, topN, item_feature_dict, item_rank_dict, fml_cat_list):
    recall = []
    recall_unexp = []
    Cov_div = []
    Cov_nov = []

    for index in range(len(topN)):
        sumForRecall = 0
        sumForRecall_unexp = 0
        sumForCov_div = 0
        sumForCov_nov = 0
        n_user = len(GroundTruth)
        n_user_with_new = len(GroundTruth)

        for i in range(len(predictedIndices)):  # for a user,
            len_test = len(GroundTruth[i])

            if len_test != 0:
                userHit = 0
                category_dict = {}
                novel_category_dict = {}

                for j in range(topN[index]):

                    unexp = False

                    for element in item_feature_dict[predictedIndices[i][j]]:
                        if element not in fml_cat_list[i]:
                            unexp = True
                            category_dict[element] = 1
                            if predictedIndices[i][j] not in item_rank_dict:
                                novel_category_dict[element] = 1
                                continue
                            if item_rank_dict[predictedIndices[i][j]] > len(item_rank_dict) / 10:
                                novel_category_dict[element] = 1
                    if unexp & (predictedIndices[i][j] in GroundTruth[i]):
                        # if Hit!
                        userHit += 1
                len_unfml = 0
                for item in GroundTruth[i]:
                    for cat in item_feature_dict[item]:
                        if cat not in fml_cat_list[i]:
                            len_unfml += 1
                            break
                if len_unfml == 0:
                    n_user_with_new -= 1
                else:
                    sumForRecall_unexp += userHit / len_unfml
                sumForRecall += userHit / len_test
                sumForCov_div += len(category_dict)
                sumForCov_nov += len(novel_category_dict)
            else:
                n_user -= 1
                n_user_with_new -= 1
        # print(n_user,n_user_with_new)
        recall.append(round(sumForRecall / n_user, 4))
        recall_unexp.append(round(sumForRecall_unexp / n_user_with_new, 4))
        Cov_div.append(round(sumForCov_div / n_user, 4))
        Cov_nov.append(round(sumForCov_nov / n_user, 4))
    return recall, recall_unexp, Cov_div, Cov_nov


def computeEntGini(GroundTruth, predictedIndices, topN, item_feature_dict, item_rank_dict, category_num):
    entro = []
    gini = []
    n_user = len(GroundTruth)
    for index in range(len(topN)):
        sumEnt = 0
        sumGini = 0
        for i in range(len(predictedIndices)):  # for a user,
            len_test = len(GroundTruth[i])
            if len_test != 0:
                category_dict = {}
                for j in range(topN[index]):
                    for element in item_feature_dict[predictedIndices[i][j]]:
                        if element in category_dict:
                            category_dict[element] += 1
                        else:
                            category_dict[element] = 1
                cat = np.array(list(category_dict.values()))
                # print(cat)
                ent = entropy(cat)
                # print(ent)
                sumEnt += ent

                count = np.sort(cat)
                n = len(count)
                cum_count = np.cumsum(count)

                sumGini += (n + 1 - 2 * np.sum(cum_count) / cum_count[-1]) / n

        entro.append(round(sumEnt / n_user, 4))
        gini.append(round(sumGini / n_user, 4))
    return entro, gini


def print_results(valid_result, test_result):
    """output the evaluation results."""
    # recall, NDCG, Cov_div, Cov_nov
    if valid_result is not None:
        print("[Valid]: recall: {} NDCG: {} Cov_div: {} Cov_nov: {} Cov_pos: {}  recall_unpop: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]]),
            '-'.join([str(x) for x in valid_result[4]]),
            '-'.join([str(x) for x in valid_result[5]])))
    if test_result is not None:
        print("[Test]: recall: {} NDCG: {} Cov_div: {} Cov_nov: {} Cov_pos: {} recall_unpop: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]]),
            '-'.join([str(x) for x in test_result[4]]),
            '-'.join([str(x) for x in test_result[5]])))


def print_results_unexp(valid_result, test_result):
    """output the evaluation results."""
    # recall, Cov_div, Cov_nov
    if valid_result is not None:
        print("[Valid]: hit_unexp: {} recall_unexp:{} Cov_div_unexp: {} Cov_nov_unexp: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]])))

    if test_result is not None:
        print("[Test]: hit_unexp: {} recall_unexp:{} Cov_div_unexp: {} Cov_nov_unexp: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]])))


def print_ent_gini(valid_result, test_result):
    """output the evaluation results."""
    # recall, Cov_div, Cov_nov
    if valid_result is not None:
        print("[Valid]: entropy: {} gini: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]])))
    if test_result is not None:
        print("[Test]: entropy: {} gini: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]])))