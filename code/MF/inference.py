import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import config
import evaluate
import data_utils
from tqdm import tqdm
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",
	type=int,
	default=30,
	help="training epoches")
parser.add_argument("--lr",
	type=float,
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout",
	type=float,
	default=0.0,
	help="dropout rate")
parser.add_argument("--batch_size",
	type=int,
	default=16,
	help="batch size for training")
parser.add_argument("--top_k",
	type=int,
	default=[10,20,50,100],
	help="compute metrics@top_k")
parser.add_argument("--factor_num",
	type=int,
	default=512,
	help="predictive factors numbers in the model")
parser.add_argument("--model_path",
    type=str,
    default="./models/",
    help="saved model path")
parser.add_argument("--num_ng",
	type=int,
	default=24,
	help="number of negtive samples")
parser.add_argument("--dataset",
    type=str,
    default="kuai",
    help="dataset:['ml-1m', 'amazon', 'kuai','kuairand','yahoo','coat']")
parser.add_argument("--model_name",
    type=str,
    default="None",
    help="exact model name")
args = parser.parse_args()

cudnn.benchmark = True

if __name__ == '__main__':
    train_data, test_data, valid_data, user_num ,item_num, train_mat, train_dic = data_utils.load_all()
    if args.model_name == "None":
        model = torch.load(args.model_path + 'MF_{}_{}fn_{}lr_{}bs_{}ng_{}un.pth'.format(args.dataset,args.factor_num,args.lr,args.batch_size,args.num_ng, user_num))
    else:
        model = torch.load(args.model_path + args.model_name)
    model.eval()
    embed_user, embed_item = model.get_emb()
    embed_user = embed_user.weight.detach()
    embed_item = embed_item.weight.detach()
    save_embed_item = False

    if not os.path.isdir('./embeddings'):
        os.makedirs('./embeddings')
    results_t, results_unexp_t, ent_gini_t,indices_top1k, scores_top1k = evaluate.metrics(model, 'test', train_data, item_num, user_num, valid_data, test_data,train_dic, args.top_k, return_pred=True)
    np.save('./embeddings/indices_MF_{}_{}factor_num_{}lr_{}bs_{}ng_{}un.npy'.format(
                             args.dataset, args.factor_num, args.lr, args.batch_size, args.num_ng, user_num), indices_top1k)
    np.save('./embeddings/scores_MF_{}_{}factor_num_{}lr_{}bs_{}ng_{}un.npy'.format(
                             args.dataset, args.factor_num, args.lr, args.batch_size, args.num_ng, user_num), scores_top1k)
    np.save('./embeddings/embed_user_MF_{}_{}factor_num_{}lr_{}bs_{}ng_{}un.npy'.format(
                             args.dataset, args.factor_num, args.lr, args.batch_size, args.num_ng, user_num), embed_user.cpu())
    np.save('./embeddings/embed_item_MF_{}_{}factor_num_{}lr_{}bs_{}ng_{}un.npy'.format(
                             args.dataset, args.factor_num, args.lr, args.batch_size, args.num_ng, user_num), embed_item.cpu())
    evaluate.print_results(None, results_t)
    evaluate.print_results_unexp(None, results_unexp_t)
    evaluate.print_ent_gini(None, ent_gini_t)

