import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

# from tensorboardX import SummaryWriter

random_seed = 1
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu
import random

random.seed(1)
import numpy as np

np.random.seed(1)

import model
import config
import evaluate
import data_utils
from tqdm import tqdm

parser = argparse.ArgumentParser()
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
parser.add_argument("--epochs",
                    type=int,
                    default=30,
                    help="training epoches")
parser.add_argument("--top_k",
                    type=int,
                    default=[10, 20, 50, 100],
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=512,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
                    type=int,
                    default=3,
                    help="number of layers in MLP model")
parser.add_argument("--num_ng",
                    type=int,
                    default=24,
                    help="sample negative items for training")
parser.add_argument("--test_num_ng",
                    type=int,
                    default=99,
                    help="sample part of negative items for testing")
parser.add_argument("--out",
                    default=True,
                    help="save model or not")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
parser.add_argument("--model_path",
                    type=str,
                    default="./models/",
                    help="saved model path")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

if __name__ == '__main__':

    ############################## PREPARE DATASET ##########################
    train_data, test_data, valid_data, user_num, item_num, train_mat, train_dic = data_utils.load_all()
    # os.system("pause")
    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(
        test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size, shuffle=True, num_workers=4)
    ########################### CREATE MODEL #################################
    GMF_model = None
    MLP_model = None

    model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                      args.dropout, config.model, GMF_model, MLP_model)
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()
    # Sigmoid 层和 BCELoss 层

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # adaptive moment estimation
    # writer = SummaryWriter() # for visualizationa

    ########################### TRAINING #####################################
    count, best_recall, best_epoch = 0, 0.0, 0
    for epoch in range(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in (train_loader):
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()
            label = label.float()
            model.zero_grad()
            prediction = model(user, item)
            # print(prediction)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        if epoch % 2 == 0:
            results, results_unexp, ent_gini = evaluate.metrics(model, 'val', train_data, item_num, user_num,
                                                                valid_data, valid_data, train_dic, args.top_k)
            elapsed_time = time.time() - start_time
            print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
                  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
            evaluate.print_results(results, None)
            recall = results[0][0]
            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch
                if not os.path.exists(args.model_path): os.mkdir(args.model_path)
                torch.save(model,
                           args.model_path + 'MF_{}_{}fn_{}lr_{}bs_{}ng_{}un.pth'.format(config.dataset, args.factor_num,
                                                                                    args.lr, args.batch_size,
                                                                                    args.num_ng, user_num))
                results_t, results_unexp_t, ent_gini_t = evaluate.metrics(model, 'test', train_data, item_num, user_num,
                                                                          valid_data, test_data, train_dic, args.top_k)
                evaluate.print_results(None, results_t)
                evaluate.print_results_unexp(None, results_unexp_t)
                evaluate.print_ent_gini(None, ent_gini_t)

    print("End. Best epoch {:03d}".format(best_epoch))
