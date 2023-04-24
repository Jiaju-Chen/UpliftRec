#!/usr/local/bin/python
from evaluate import *
from util import *
import os
import time
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import math

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default='kuai',
                    help="dataset")
parser.add_argument("--top_k",
                    type=str,
                    default='[10]',
                    help="compute metrics@top_k")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
parser.add_argument("--data_path",
                    type=str,
                    default="../../data/kuai/kuai-v1/",
                    help="data path for evaluation")
parser.add_argument("--subdata_path",
                    type=str,
                    default="../../data/kuai/kuai-subusers-cluster/",
                    help="data path for subusers")
parser.add_argument("--backend_path",
                    type=str,
                    default="../MMR/data/MF/",
                    help="data path for backend model")
parser.add_argument("--backend_modelname",
                    type=str,
                    default="_MF_kuai_256factor_num_0.0001lr_1024bs_0.2dropout.npy",
                    help="name of backend model")
parser.add_argument("--embed_file",
                    type=str,
                    default="embed_user_MF_kuai_256factor_num_0.0001lr_1024bs_0.2dropout.npy",
                    help="name of all-user embedding file")
parser.add_argument("--similar_user_num",
                    type=int,
                    default=100,
                    help="similar user number for ADRF")
parser.add_argument("--similar_user_num_propensity",
                    type=int,
                    default=100,
                    help="similar user number for propensity")
parser.add_argument("--treat_clip_num",
                    type=int,
                    default=10,
                    help="treat clip number")
parser.add_argument("--ADRF_null",
                    type=float,
                    default=0.01,
                    help="padding value in ADRF matrix")
parser.add_argument("--propensity_null",
                    type=float,
                    default=0.01,
                    help="padding value in propensity matrix")
parser.add_argument("--MTEF_null",
                    type=float,
                    default=0.0,
                    help="padding value in MTEF if data is not complete")
parser.add_argument("--use_MLP",
                    type=int,
                    default=0,
                    help="whether to use MLP or simply use statistical treatment data.")
                    #  We suggest the statistical method only.
parser.add_argument("--MLP_path",
                    type=str,
                    default='./MLP/models/',
                    help="data path for MLP model")
parser.add_argument("--MLP_propensity_model",
                    type=str,
                    default='MLP.pth',
                    help="name of propensity model")
parser.add_argument("--eps",
                    type=int,
                    default=1,
                    help="eps for DP_solver")
parser.add_argument("--delta_T",
                    type=int,
                    default=1,
                    help="delta_T for MTEF")
parser.add_argument("--use_MTEF",
                    type=int,
                    default=1,
                    help="whether to use MTEF or ADRF")
parser.add_argument("--alpha_MTEF",
                    type=float,
                    default=0.0,
                    help="the weight of the MTEF adjustment")
parser.add_argument("--check_user",
                    type=int,
                    default=-1,
                    help="print this user's ADRF, T0, and other data")
parser.add_argument("--MLP_drop",
                    type=float,
                    default=0.8,
                    help="dropout for MLP")
parser.add_argument("--use_conv_ADRF",
                    type=int,
                    default=1,
                    help="use convolutional kernels to make ADRF smooth")
                    # we find it useless in Yahoo!R3, Coat and KuaiRec
parser.add_argument("--conv_num_ADRF",
                    type=int,
                    default=0,
                    help="number of convolutions")
parser.add_argument("--ker0",
                    type=str,
                    default='[0.8,0.2]',
                    help="kernel on the edge")
parser.add_argument("--ker1",
                    type=str,
                    default='[0.1,0.8,0.1]',
                    help="ker in the middle")
parser.add_argument("--use_onehot_embed",
                    type=int,
                    default=1,
                    help="calculate the users' similarity based only on the number of their overlapping history items")
parser.add_argument("--use_topk",
                    type=int,
                    default=100,
                    help="topk indices from the backend model used to re-rank in UpliftRec-MTEF")
parser.add_argument("--gamma",
                    type=float,
                    default=1,
                    help="propensity reweight parameter")


args = parser.parse_args()
print(args)
top_k = eval(args.top_k)
treat_clip_num = args.treat_clip_num
ADRF_null = args.ADRF_null
propensity_null = args.propensity_null
eps = args.eps
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cudnn.benchmark = True

#### files for evaluation
# the item features used when estimating the final recommendation list
item_feature_dict = np.load(args.data_path + 'features/item_feature_dict.npy', allow_pickle=True).item()
# the item features used when calculating the ADRF matrix and make recommendations
item_feature_dict_cluster = np.load(args.subdata_path + 'features/item_feature_dict.npy', allow_pickle=True).item()
# the items' popularity rank
item_rank_dict = np.load(args.data_path + 'features/item_rank_dict.npy', allow_pickle=True).item()
# familiar categories for each user
user_fml_cat = np.load(args.data_path + 'features/user_fml_cat.npy', allow_pickle=True).item()
# dataset
train_data = np.load(args.data_path + 'sets/training_dict.npy', allow_pickle=True).item()
valid_data = np.load(args.data_path + 'sets/validation_dict.npy', allow_pickle=True).item()
test_data = np.load(args.data_path + 'sets/testing_dict.npy', allow_pickle=True).item()
# save familiar categories in a list
fml_cat_list = []
for user in valid_data:
    fml_cat_list.append(user_fml_cat[user])
# print number of items and users
item_num = max(item_rank_dict.keys()) + 1
print("item_num:", item_num)
user_num = max(train_data.keys()) + 1
print("user_num:", user_num)

# subuser file which contain each subuser's D, T, Y
user_info = np.load(args.subdata_path + 'features/subuser_info.npy', allow_pickle=True).item()
# generate the embedding for similarity
if args.use_onehot_embed:
    users_onehot_embed = torch.zeros(len(user_info) + user_num, item_num)
    for subu in user_info:
        D = user_info[subu]['D']
        users_onehot_embed[subu][D] = 1
    for u in train_data:
        D = train_data[u]
        users_onehot_embed[u][D] = 1
    cnt_onehot_all = users_onehot_embed @ users_onehot_embed[user_num:].T  # all users' similarity to subusers
    cnt_onehot_sub = cnt_onehot_all[user_num:]
    cnt_onehot_ori = cnt_onehot_all[:user_num]
    print("all user_num:", cnt_onehot_all.shape[0])
    assert cnt_onehot_all.shape[0] - len(user_info) == user_num
else:
    user_embeddings0 = np.load(args.backend_path +  args.embed_file)
    user_embeddings = np.zeros([user_num + len(user_info), user_embeddings0.shape[1]])
    user_embeddings[:user_num] = user_embeddings0[:user_num]
    user_embeddings[-len(user_info):] = user_embeddings0[-len(user_info):]
    print("all user_num:", user_embeddings.shape[0])
    for i in range(user_embeddings.shape[0]):
        user_embeddings[i] /= np.linalg.norm(user_embeddings[i])
    assert user_embeddings.shape[0] - len(user_info) == user_num
cate_num = len(user_info[user_num]['Y'])
print("cate_num:", cate_num)

data_generated_path ='./data_generated/'
if not os.path.isdir(data_generated_path):
    os.makedirs(data_generated_path)

# calculate propensity
t1 = time.time()
user_propensity = {}
propensity_path = '{}{}_{}Pn_{}tn_{}cn_{}emb_propensity.npy'.format(data_generated_path, args.dataset, args.similar_user_num_propensity, treat_clip_num, cate_num, args.embed_file)
if args.use_MLP: # we do not suggest MLP
    pass
elif args.use_onehot_embed:
    if os.path.exists(propensity_path):
        user_propensity = np.load(propensity_path, allow_pickle=True).item()
        print("user_propensity loaded")
    else:
        _, indices = torch.topk(cnt_onehot_sub, args.similar_user_num_propensity)
        indices += user_num
        indices = indices.tolist()
        for user in user_info:
            user_propensity[user] = np.full([cate_num, treat_clip_num + 1], 0.0)
            count_list = [0] * treat_clip_num
            for u in indices[user - user_num]:
                for c in range(cate_num):
                    user_treat = treatment2index(user_info[u]['T'][c], treat_clip_num)
                    user_propensity[user][c][user_treat] += 1
            for c in range(cate_num):
                user_propensity[user][c] /= user_propensity[user][c].sum()
            for c in range(cate_num):
                for treat in range(treat_clip_num + 1):
                    if user_propensity[user][c][treat] < args.propensity_null: user_propensity[user][c][
                        treat] = args.propensity_null
        np.save(propensity_path, np.array(user_propensity))
else: # use embedding
    if os.path.exists(propensity_path):
        user_propensity = np.load(propensity_path, allow_pickle=True).item()
        print("user_propensity loaded")
    else:
        for user in user_info:
            user_propensity[user] = np.full([cate_num, treat_clip_num + 1], 0.0)
            user_embed = user_embeddings[user]
            user_similarity = np.dot(user_embeddings[user_num:], user_embed.T)  #
            top_K_users = np.argsort(-user_similarity)[:args.similar_user_num_propensity] + user_num
            count_list = [0] * treat_clip_num
            for u in top_K_users:
                for c in range(cate_num):
                    user_treat = treatment2index(user_info[u]['T'][c], treat_clip_num)
                    user_propensity[user][c][user_treat] += 1
            for c in range(cate_num):
                user_propensity[user][c] /= user_propensity[user][c].sum()
            for c in range(cate_num):
                for treat in range(treat_clip_num + 1):
                    if user_propensity[user][c][treat] < args.propensity_null:
                        user_propensity[user][c][treat] = args.propensity_null
        np.save(propensity_path, np.array(user_propensity))
print('calculating propensity costs', time.time() - t1, 's')

# get backend scores
t2 = time.time()
indices_dic = np.load(args.backend_path + "indices" + args.backend_modelname, allow_pickle=True).item()
scores_dic = np.load(args.backend_path + "scores" + args.backend_modelname, allow_pickle=True).item()

# calculate scores
if args.use_MTEF == 1: # for UpliftRec-MTEF
    data_path = '{}{}_user_ranking_{}top_{}cn.npy'.format(data_generated_path, args.dataset, args.use_topk, cate_num)
    if os.path.exists(data_path):
        user_ranking = np.load(data_path, allow_pickle=True).item()
        print("user_ranking loaded")
    else:
        user_ranking = {}
        for user in indices_dic:
            indices = indices_dic[user].cpu().tolist()[:args.use_topk]
            scores = scores_dic[user].cpu().tolist()[:args.use_topk]
            user_ranking[user] = [[] for c in range(cate_num)]
            for i, item in enumerate(indices):
                for c in item_feature_dict_cluster[item]:
                    user_ranking[user][c].append([item, scores[i]])
        np.save(data_path, np.array(user_ranking))
else: # for UpliftRec-ADRF
    data_path = '{}{}_user_ranking_all_{}cn.npy'.format(data_generated_path, args.dataset, cate_num)
    if os.path.exists(data_path):
        user_ranking = np.load(data_path, allow_pickle=True).item()
        print("user_ranking loaded")
    else:
        user_ranking = {}
        for user in indices_dic:
            indices = indices_dic[user].cpu().tolist()
            scores = scores_dic[user].cpu().tolist()
            user_ranking[user] = [[] for c in range(cate_num)]
            for i, item in enumerate(indices):
                for c in item_feature_dict_cluster[item]:
                    user_ranking[user][c].append([item, scores[i]])
        np.save(data_path, np.array(user_ranking))
print('calculating user_ranking costs', time.time() - t2, 's')


# start UpliftRec
user_rec_dict = {}
GroundTruth = []
predictedIndices = {}
T0_all = np.zeros(cate_num)
for n in top_k: predictedIndices[n] = []
if args.use_onehot_embed:
    _, indices_ori = torch.topk(cnt_onehot_ori, args.similar_user_num)
    indices_ori += user_num
    indices_ori = indices_ori.tolist()
for user in (test_data):
    # step 1 find similar users.
    # get user embedding and calc similarity: only calculate sim with subusers with T and Y
    if args.use_onehot_embed:
        top_K_users = indices_ori[user]
    else:
        user_embed = user_embeddings[user]
        user_similarity = np.dot(user_embeddings[user_num:], user_embed.T)
        # find top K1 similar users: +user_num is to shift id
        top_K_users = np.argsort(-user_similarity)[:args.similar_user_num] + user_num

    # step 2 calculate ADRF.
    # obtain category num
    ADRF = np.ones([cate_num, treat_clip_num + 1], dtype=float) * ADRF_null

    for c in range(cate_num):
        # record Y and normalization weight n
        y_pre = [0] * (treat_clip_num + 1)
        n_pre = [0] * (treat_clip_num + 1)

        for u in top_K_users:
            user_treat = treatment2index(user_info[u]['T'][c], treat_clip_num)
            y_pre[user_treat] += user_info[u]['Y'][c] / math.pow(user_propensity[u][c][user_treat], args.gamma)
            n_pre[user_treat] += 1 / math.pow(user_propensity[u][c][user_treat], args.gamma)
        y_norm = [y_pre[t] / n_pre[t] if n_pre[t] > 0 else 0 for t in range(treat_clip_num + 1)]

        # update ADRF
        for k in range(treat_clip_num + 1):
            if y_norm[k] > 0:
                ADRF[c, k] = y_norm[k]
        ADRF[c, 0] = 0
    if user == args.check_user: print("ADRF0", ADRF)

    if args.use_conv_ADRF:
        ker0 = eval(args.ker0)
        ker1 = eval(args.ker1)
        for i in range(args.conv_num_ADRF):
            ADRF_new = np.ones([cate_num, treat_clip_num + 1], dtype=float) * 0
            for c in range(cate_num):
                ADRF_new[c, 1] = ADRF[c, 1] * ker0[0] + ADRF[c, 2] * ker0[1]
                for t in range(2, treat_clip_num):
                    ADRF_new[c, t] = ADRF[c, t - 1] * ker1[0] + ADRF[c, t] * ker1[1] + ADRF[c, t + 1] * ker1[2]
                ADRF_new[c, treat_clip_num] = ADRF[c, treat_clip_num] * ker0[0] + ADRF[c, treat_clip_num - 1] * ker0[1]
            ADRF = ADRF_new
            if user == args.check_user: print("ADRF_{}".format(i), ADRF)

    # step 3 fetch the backend recommendation list.
    Top_ori = indices_dic[user][:top_k[-1]].cpu().tolist()
    T_continuous = calc_T(Top_ori, item_feature_dict_cluster, cate_num)
    T0 = [treatment2index(t, treat_clip_num) for t in T_continuous]
    T0_all += np.array(T0)
    if user == args.check_user:
        print("Top_ori", Top_ori)
        print("T_continuous", T_continuous)
        print("T0", T0)

    # step 4 MTEF calculation and recommend
    if args.use_MTEF:
        MTEF = [0] * cate_num
        for c in range(cate_num):
            if (T0[c] + args.delta_T > treat_clip_num) or (ADRF[c][T0[c] + args.delta_T] == args.ADRF_null) or (
                    ADRF[c][T0[c]] == args.ADRF_null):
                MTEF[c] = args.MTEF_null
            else:
                MTEF[c] = (ADRF[c][T0[c] + args.delta_T] - ADRF[c][T0[c]]) / args.delta_T
        if user == args.check_user: print('MTEF:', MTEF)
        user_rank = user_ranking[user]
        for n in top_k:
            if user not in user_rec_dict:
                user_rec_dict[user] = {}
            rec_list = []
            for c in range(cate_num):
                for i in user_rank[c]:
                    rec_list.append([i[0], i[1] + args.alpha_MTEF * MTEF[c]])

            rec_list.sort(key=rank_second, reverse=True)
            rec_list = [item[0] for item in rec_list]
            for index, item_tuple in enumerate(rec_list):
                if item_tuple in rec_list[index + 1:]:
                    rec_list.remove(item_tuple)
            rec_list = rec_list[:n]
            user_rec_dict[user][n] = rec_list
            predictedIndices[n].append(user_rec_dict[user][n])
            if user == args.check_user: print('rec_list', n, rec_list)
        if user == args.check_user: print('GroundTruth', test_data[user])
        GroundTruth.append(test_data[user])
    # step 4 best treatment calculation and recommend
    else:
        _, best_T = DP_solve(cate_num, treat_clip_num, ADRF, T0, eps)
        if user == args.check_user: print('T0', T0)
        if user == args.check_user: print('best_T', best_T)
        if sum(best_T) != 1:
            best_T = best_T / sum(best_T)

        for n in top_k:
            if user not in user_rec_dict:
                user_rec_dict[user] = {}

            # decide the rec num of each category
            rec_num = [int(n * t) for t in best_T]
            if sum(rec_num) < n:
                res = np.random.choice(range(cate_num), n - sum(rec_num), replace=True, p=best_T)
                for x in res:
                    rec_num[x] += 1
            rec_list = []
            user_rank = user_ranking[user]
            for c in range(cate_num):
                rec_list += (user_rank[c][:rec_num[c]])
            # process: remove same ids. rec_list
            for index, item_tuple in enumerate(rec_list):
                if item_tuple in rec_list[index + 1:]:
                    cat_index = np.random.randint(len(item_feature_dict_cluster[item_tuple[0]]))
                    cat = item_feature_dict_cluster[item_tuple[0]][cat_index]
                    for item_replace in user_rank[cat]:
                        if item_replace not in rec_list:
                            rec_list[index] = item_replace
                    # find_next tuple of that category and replace
            rec_list.sort(key=rank_second, reverse=True)
            user_rec_dict[user][n] = [item[0] for item in rec_list]
            predictedIndices[n].append(user_rec_dict[user][n])
            if user == args.check_user: print('rec_list', n, user_rec_dict[user][n])
        GroundTruth.append(test_data[user])
        if user == args.check_user: print('GroundTruth', test_data[user])
print('T0_all:', T0_all)

# evaluate
for n in top_k:
    results = computeTopNAccuracy(GroundTruth, predictedIndices[n], [n], item_feature_dict, item_rank_dict)
    results_unexp = computeUnexp(GroundTruth, predictedIndices[n], [n], item_feature_dict, item_rank_dict, fml_cat_list)
    e_g = computeEntGini(GroundTruth, predictedIndices[n], [n], item_feature_dict, item_rank_dict, cate_num)
    print_results(None, results)
    print_results_unexp(None, results_unexp)
    print_ent_gini(None, e_g)
