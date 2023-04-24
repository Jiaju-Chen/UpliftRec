import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils

import random

random_seed = 1
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default="yahoo",
                    help="dataset option: 'ml_1m', 'amazon'")
parser.add_argument("--model",
                    type=str,
                    default="FM",
                    help="model option: 'NFM' or 'FM'")
parser.add_argument("--loss_type",
                    type=str,
                    default="log_loss",
                    help="loss option: 'square_loss' or 'log_loss'")
parser.add_argument("--optimizer",
                    type=str,
                    default="Adagrad",
                    help="optimizer option: 'Adagrad', 'Adam', 'SGD', 'Momentum'")
parser.add_argument("--data_path",
                    type=str,
                    default="../../data/yahoo/yahoo-v1/",
                    help="load data path")
parser.add_argument("--model_path",
                    type=str,
                    default="./models/",
                    help="saved model path")
parser.add_argument("--activation_function",
                    type=str,
                    default="relu",
                    help="activation_function option: 'relu', 'sigmoid', 'tanh', 'identity'")
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="learning rate")
parser.add_argument("--dropout",
                    default='[0.4,0.2]',
                    help="dropout rate for FM and MLP")
parser.add_argument("--batch_size",
                    type=int,
                    default=1024,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=80,
                    help="training epochs")
parser.add_argument("--hidden_factor",
                    type=int,
                    default=512,
                    help="predictive factors numbers in the model")
parser.add_argument("--layers",
                    default='[64]',
                    help="size of layers in MLP model, '[]' is NFM-0")
parser.add_argument("--lamda",
                    type=float,
                    default=0.0,
                    help="regularizer for bilinear layers")
parser.add_argument("--topN",
                    default='[10, 20, 50, 100]',
                    help="the recommended item num")
parser.add_argument("--batch_norm",
                    type=int,
                    default=1,
                    help="use batch_norm or not. option: {1, 0}")
parser.add_argument("--pre_train",
                    action='store_true',
                    default=False,
                    help="whether use the pre-train or not")
parser.add_argument("--pre_train_model_path",
                    type=str,
                    default="./models/",
                    help="pre_trained model_path")
parser.add_argument("--out",
                    default=True,
                    help="save model or not")
parser.add_argument("--ood_test",
                    default=False,
                    help="whether test ood data during iid training")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
parser.add_argument('--portion',
                    type=str,
                    default='0.1',
                    help='portion of ood training'
                    )
args = parser.parse_args()
print("args:", args)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

if __name__ == '__main__':
    #############################  PREPARE DATASET #########################
    start_time = time.time()
    item_feature_dict = np.load(args.data_path + 'features/item_feature_dict.npy', allow_pickle=True).item()
    item_rank_dict = np.load(args.data_path + 'features/item_rank_dict.npy', allow_pickle=True).item()
    user_fml_cat = np.load(args.data_path + 'features/user_fml_cat.npy', allow_pickle=True).item()
    category_num = len(np.load(args.data_path + 'features/category_list.npy'))
    item_cat_vec = np.load(args.data_path + 'features/item_cat_vec.npy', allow_pickle=True).item()

    train_path = args.data_path + 'sets/training_list.npy'
    valid_path = args.data_path + 'sets/validation_dict.npy'
    test_path = args.data_path + 'sets/testing_dict.npy'

    user_feature_path = args.data_path + 'features/user_feature_file.npy'
    item_feature_path = args.data_path + 'features/item_feature_file.npy'
    user_feature, item_feature, num_features = data_utils.map_features(user_feature_path, item_feature_path)
    train_dataset = data_utils.FMData(train_path, user_feature, item_feature, args.loss_type)
    valid_dict = data_utils.loadData(valid_path)
    test_dict = data_utils.loadData(test_path)

    all_item_features, all_item_feature_values = evaluate.pre_ranking(item_feature)

    model = torch.load('{}{}_{}_{}hidden_{}layer_{}lr_{}bs_{}dropout_{}lamda_{}bn_{}_{}opt.pth'.format(
        args.model_path, args.model, args.dataset, args.hidden_factor, args.layers, args.lr,
        args.batch_size, args.dropout, args.lamda, args.batch_norm, args.loss_type, args.optimizer))

    model.eval()
    embedding_all = model.embeddings.weight.detach().cpu().numpy()
    embed_user = np.zeros([len(user_feature), args.hidden_factor])
    for user in user_feature:
        embed_user[user] = embedding_all[user_feature[user][0][0]]
    embed_item = np.zeros([len(item_feature), args.hidden_factor])
    for item in item_feature:
        embed_item[item] = embedding_all[item_feature[item][0][0]]
    cate_feature_id = item_feature[0][0][1:]
    embed_cate = np.zeros([category_num, args.hidden_factor])
    for cate in range(category_num):
        embed_cate[cate] = embedding_all[cate_feature_id[cate]]

    valid_results, test_results, valid_results_unexp, test_results_unexp, valid_ent_gini, test_ent_gini, indices_top1k, scores_top1k = evaluate.Ranking(
        model, valid_dict, test_dict, train_dataset.train_dict,
        user_feature, all_item_features, all_item_feature_values, item_feature_dict,
        item_rank_dict, 2048, eval(args.topN), user_fml_cat, category_num, item_cat_vec, return_pred=True)

    np.save('./embeddings/indices_FM_{}_{}hidden_factor{}lr_{}bs_{}dropout.npy'.format(
        args.dataset, args.hidden_factor, args.lr, args.batch_size, args.dropout), indices_top1k)
    np.save('./embeddings/scores_FM_{}_{}hidden_factor{}lr_{}bs_{}dropout.npy'.format(
        args.dataset, args.hidden_factor, args.lr, args.batch_size, args.dropout), scores_top1k)
    np.save('./embeddings/embed_user_FM_{}_{}hidden_factor{}lr_{}bs_{}dropout.npy'.format(
        args.dataset, args.hidden_factor, args.lr, args.batch_size, args.dropout), embed_user)
    np.save('./embeddings/embed_item_FM_{}_{}hidden_factor{}lr_{}bs_{}dropout.npy'.format(
        args.dataset, args.hidden_factor, args.lr, args.batch_size, args.dropout), embed_item)
    np.save('./embeddings/embed_cate_FM_{}_{}hidden_factor{}lr_{}bs_{}dropout.npy'.format(
        args.dataset, args.hidden_factor, args.lr, args.batch_size, args.dropout), embed_cate)

    evaluate.print_results(None, test_results)
    evaluate.print_results_unexp(None, test_results_unexp)
    evaluate.print_ent_gini(None, test_ent_gini)
