import os
import time
import argparse
import numpy as np
from tqdm import tqdm
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
                    default="coat",
                    help="dataset option: 'kuai','yahoo','coat'")
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
                    default="../../data/kuai/kuai-FM/",
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
                    default=0.05,
                    help="learning rate")
parser.add_argument("--dropout",
                    default='[0.5, 0.2]',
                    help="dropout rate for FM and MLP")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=300,
                    help="training epochs")
parser.add_argument("--hidden_factor",
                    type=int,
                    default=64,
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

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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

    train_loader = data.DataLoader(train_dataset, drop_last=True,
                                   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                   worker_init_fn=worker_init_fn)
    all_item_features, all_item_feature_values = evaluate.pre_ranking(item_feature)

    print('data ready. costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))

    ##############################  CREATE MODEL ###########################
    if args.pre_train and args.model == 'NFM':
        assert os.path.exists(args.pre_train_model_path), 'lack of FM model'
        assert args.model == 'NFM', 'only support NFM for now'
        FM_model = torch.load(args.pre_train_model_path)
    else:
        FM_model = None

    if args.model == 'FM':
        if args.pre_train:  # pre-trained model on iid
            model = torch.load(args.pre_train_model_path)
        else:
            model = model.FM(num_features, args.hidden_factor,
                             args.batch_norm, eval(args.dropout))
    elif args.model == 'NFM':
        model = model.NFM(num_features, args.hidden_factor,
                          args.activation_function, eval(args.layers),
                          args.batch_norm, eval(args.dropout), FM_model)
    else:
        raise Exception('model not implemented!')

    model.cuda()
    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

    if args.loss_type == 'square_loss':
        criterion = nn.MSELoss(reduction='sum')
    elif args.loss_type == 'log_loss':
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        raise Exception('loss type not implemented!')

    # writer = SummaryWriter() # for visualization

    ###############################  TRAINING ############################
    count, best_recall, best_epoch = 0, -100, -1
    best_test_results, best_test_results_unexp, best_test_ent_gini = None, None, None
    for epoch in range(args.epochs):
        # print(epoch)
        model.train()  # Enable dropout and batch_norm
        start_time = time.time()
        train_loader.dataset.ng_sample()
        #     print('sampling done. costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))

        for features, feature_values, label in train_loader:
            features = features.cuda()
            feature_values = feature_values.cuda()
            label = label.cuda()
            # print(label)
            model.zero_grad()
            prediction = model(features, feature_values)
            loss = criterion(prediction, label)
            loss += args.lamda * model.embeddings.weight.norm()
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        if epoch % 2 == 0:
            #     if epoch > 100 and epoch % 10 == 0:
            model.eval()
            train_RMSE = evaluate.RMSE(model, args.model, train_loader)
            valid_results, test_results, valid_results_unexp, test_results_unexp, valid_ent_gini, test_ent_gini = evaluate.Ranking(
                model, valid_dict, test_dict, train_dataset.train_dict, \
                user_feature, all_item_features, all_item_feature_values, item_feature_dict, \
                item_rank_dict, 40000, eval(args.topN), user_fml_cat, category_num, item_cat_vec)
            print('---' * 18)
            print("Runing Epoch {:03d} ".format(epoch) + 'loss {:.4f}'.format(loss) + " costs " + time.strftime(
                "%H: %M: %S", time.gmtime(time.time() - start_time)))
            evaluate.print_results(valid_results, test_results)

            if valid_results[0][0] > best_recall:  # recall@10 for selection
                best_recall, best_epoch = valid_results[0][0], epoch
                best_test_results = test_results
                best_test_results_unexp = test_results_unexp
                best_test_ent_gini = test_ent_gini
                print("------------Best model, saving...------------")
                if args.out:
                    if not os.path.exists(args.model_path):
                        os.mkdir(args.model_path)
                    if 1:
                        torch.save(model,
                                   '{}{}_{}_{}hidden_{}layer_{}lr_{}bs_{}dropout_{}lamda_{}bn_{}_{}opt.pth'.format(
                                       args.model_path, args.model, args.dataset, args.hidden_factor, args.layers,
                                       args.lr, args.batch_size, args.dropout, args.lamda, args.batch_norm, args.loss_type,
                                       args.optimizer))

    print("End. Best epoch {:03d}".format(best_epoch))
    evaluate.print_results(None, best_test_results)
    evaluate.print_results_unexp(None, best_test_results_unexp)
    evaluate.print_ent_gini(None, best_test_ent_gini)




