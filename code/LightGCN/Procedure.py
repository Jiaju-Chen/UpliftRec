import world
import numpy as np
import torch
import utils
import dataloader
import evaluate
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def Test(dataset, Recmodel, epoch, w=None, multicore=0, return_pred=False):
    data_path = world.config['data_path'] + '/'
    item_feature_dict = np.load(data_path + 'item_feature_dict.npy', allow_pickle=True).item()
    item_rank_dict = np.load(data_path + 'item_rank_dict.npy', allow_pickle=True).item()
    category_num = len(np.load(data_path + 'category_list.npy'))
    user_fml_cat = np.load(data_path + 'user_fml_cat.npy', allow_pickle=True).item()

    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset

    trainDict = np.load(data_path + 'training_dict.npy', allow_pickle=True).item()
    valDict = np.load(data_path + 'validation_dict.npy', allow_pickle=True).item()
    testDict = np.load(data_path + 'testing_dict.npy', allow_pickle=True).item()

    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    with torch.no_grad():
        # valid
        #results = np.array([[0.0 for i in range(len(world.topks))] for j in range(6)])
        #results_unexp = np.array([[0.0 for i in range(len(world.topks))] for j in range(4)])
        #ent_gini = np.array([[0.0 for i in range(len(world.topks))] for j in range(2)])

        users = list(valDict.keys())
#         try:
#             assert u_batch_size <= len(users) / 10
#         except AssertionError:
#             print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
#         users_list = []
#         rating_list = []
#         groundTrue_list = []
        # auc_record = []
        # ratings = []
        #total_batch = len(users) // u_batch_size + 1
        #for batch_users in utils.minibatch(users, batch_size=u_batch_size):
        batch_users = users
        if 1:
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [valDict[u] for u in batch_users]
            fml_cat_list = [user_fml_cat[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            #users_list.append(batch_users)
            rating_K = rating_K.tolist()
            #print(np.mean([sum(i in groundTrue[j] for i in rating_K[j]) / len(groundTrue[j]) for j in range(len(rating_K))]))

            results = evaluate.computeTopNAccuracy(groundTrue, rating_K, world.topks, item_feature_dict,
                                                    item_rank_dict)
            results_unexp = evaluate.computeUnexp(groundTrue, rating_K, world.topks, item_feature_dict, item_rank_dict,
                                                   fml_cat_list)
            ent_gini = evaluate.computeEntGini(groundTrue, rating_K, world.topks, item_feature_dict, item_rank_dict,
                                                category_num)

        # assert total_batch == len(users_list)

#         results = np.around((results / len(users)), 4).tolist()
#         results_unexp = np.around((results_unexp / len(users)), 4).tolist()
#         ent_gini = np.around((ent_gini / len(users)), 4).tolist()
        valid_results = results
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            pass
        if multicore == 1:
            pool.close()
        evaluate.print_results(results, None)
        evaluate.print_results_unexp(results_unexp, None)
        evaluate.print_ent_gini(ent_gini, None)
        ret = results

        # test
#         results = np.array([[0.0 for i in range(len(world.topks))] for j in range(6)])
#         results_unexp = np.array([[0.0 for i in range(len(world.topks))] for j in range(4)])
#         ent_gini = np.array([[0.0 for i in range(len(world.topks))] for j in range(2)])

        users = list(testDict.keys())
#         try:
#             assert u_batch_size <= len(users) / 10
#         except AssertionError:
#             print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
#         users_list = []
#         rating_list = []
        #groundTrue_list = []
        # auc_record = []
        # ratings = []
        indices_top1k = {}
        scores_top1k = {}
        total_batch = len(users) // u_batch_size + 1
        batch_users = users
        #for batch_users in utils.minibatch(users, batch_size=u_batch_size):
        if 1:
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            fml_cat_list = [user_fml_cat[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            rating = rating.cpu()
            for i in range(len(batch_users)):
                user = batch_users[i]
                for j in trainDict[user]:
                    rating[i, j] = -999
                for j in valDict[user]:
                    rating[i, j] = -999

            _, rating_K = torch.topk(rating, k=max_K)

            scores, indices = torch.topk(rating, min(1000, len(item_feature_dict) // 2))
            scores = torch.sigmoid(scores.detach())
            for i in range(len(batch_users)):
                user = batch_users[i]
                indices_top1k[user] = indices[i]
                scores_top1k[user] = scores[i]

            # rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating

            # users_list.append(batch_users)
            rating_K = rating_K.tolist()
            results = evaluate.computeTopNAccuracy(groundTrue, rating_K, world.topks, item_feature_dict,
                                                    item_rank_dict)
            results_unexp = evaluate.computeUnexp(groundTrue, rating_K, world.topks, item_feature_dict, item_rank_dict,
                                                   fml_cat_list)
            ent_gini = evaluate.computeEntGini(groundTrue, rating_K, world.topks, item_feature_dict, item_rank_dict,
                                                category_num)

        #assert total_batch == len(users_list)

#         results = np.around((results / len(users)), 4).tolist()
#         results_unexp = np.around((results_unexp / len(users)), 4).tolist()
#         ent_gini = np.around((ent_gini / len(users)), 4).tolist()

        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            pass
        if multicore == 1:
            pool.close()
        evaluate.print_results(None, results)
        evaluate.print_results_unexp(None, results_unexp)
        evaluate.print_ent_gini(None, ent_gini)
        if return_pred:
            return results, indices_top1k, scores_top1k
        return valid_results
