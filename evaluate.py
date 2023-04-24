import numpy as np
import torch
import math
from scipy.stats import entropy


def computeTopNAccuracy(GroundTruth, predictedIndices, topN, item_feature_dict, item_rank_dict):
    '''
    Parameters:
        GroundTruth: GroundTruth items for each user. if the ground truth of user u,v is 1,2 and 3,4,5,
            then GroundTruth is [[1,2], [3,4,5]]
        predictedIndices: predicted items for each user. A list whose elements are lists of topN(biggest) items for each user
        topN: the length of the recommendation list. A list of lengths.
        item_feature_dict: A dict where each item is mapped to its features
        item_rank_dict: A dict where each item is mapped to its popularity rank. The most popular item rank 0.
    '''
    pop_rate = 0.1  # popular items are top pop_rate popular items.
    recall = []
    NDCG = []
    Cov_div = [] # coverage
    Cov_nov = [] # coverage of unpopular items
    Cov_pos = [] # coverage of positive test samples
    recall_unpop = [] # recall of unpopular items

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
                        category_dict[element] = 1
                        if item_rank_dict[predictedIndices[i][j]] > (len(item_rank_dict) * pop_rate):
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