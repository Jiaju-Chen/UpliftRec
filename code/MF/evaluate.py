import numpy as np
import torch
import pandas as pd
import math
import os
import config
from scipy.stats import entropy
from tqdm import tqdm


def sigmoid(x):
    s = 1 / (1 + torch.exp(-x))
    return s


def metrics(model, v_or_t, train_data, item_num, user_num, valid_data, test_data, train_dic, top_k, return_pred=False):
    item_feature_dict = np.load(config.main_path + 'features/item_feature_dict.npy', allow_pickle=True).item()
    item_rank_dict = np.load(config.main_path + 'features/item_rank_dict.npy', allow_pickle=True).item()
    category_num = len(np.load(config.main_path + 'features/category_list.npy'))
    user_fml_cat = np.load(config.main_path + 'features/user_fml_cat.npy', allow_pickle=True).item()
    fml_cat_list = []

    GroundTruth, predictedIndices = [], []
    train_data = pd.DataFrame(train_data)
    indices_top1k = {}
    scores_top1k = {}
    u, i = model.get_emb()
    u, i = u.weight, i.weight
    rating = u @ i.T
    ratings = torch.zeros(len(valid_data), item_num)
    row = 0
    for k in (valid_data):  # range(user_num)
        if k not in test_data:
            continue
        fml_cat_list.append(user_fml_cat[k])
        item_ps = test_data[k]
        item_train = train_dic[k]

        rating[k][item_train] = -999
        if v_or_t == 'test':
            rating[k][valid_data[k]] = -999
        ratings[row] = rating[k]
        row += 1
        GroundTruth.append(item_ps)

    _, indices = torch.topk(ratings, top_k[-1])
    predictedIndices = indices.cpu().tolist()
    if return_pred:
        for k in (valid_data):
            scores_, indices_ = torch.topk(rating[k], item_num)
            indices_top1k[k] = indices_.detach()
            scores_top1k[k] = sigmoid(scores_.detach())
        # print(indices_top1k)

    results = computeTopNAccuracy(GroundTruth, predictedIndices, top_k, item_feature_dict, item_rank_dict)
    results_unexp = computeUnexp(GroundTruth, predictedIndices, top_k, item_feature_dict, item_rank_dict, fml_cat_list)
    ent_gini = computeEntGini(GroundTruth, predictedIndices, top_k, item_feature_dict, item_rank_dict, category_num)
    if return_pred:
        return results, results_unexp, ent_gini, indices_top1k, scores_top1k
    return results, results_unexp, ent_gini


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
                        '''
                        if predictedIndices[i][j] not in item_rank_dict:
                            novel_category_dict[element] = 1
                            continue
                        '''
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

