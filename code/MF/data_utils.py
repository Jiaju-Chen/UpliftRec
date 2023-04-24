import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data
import math
import config


def load_all(test_num=100):
    """ We load all the three file here to save time in each epoch. """
    '''
    train_data = pd.read_csv(
        config.train_rating, 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
        '''
    train_data = np.load(config.train_rating)
    train_data = pd.DataFrame(train_data)

    user_num = train_data[0].max() + 1
    item_num = train_data[1].max() + 1
    train_data.sort_index()
    # print(train_data.groupby("user").size())
    train_data1 = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data1:
        train_mat[x[0], x[1]] = 1.0

    test_dic = np.load(config.main_path + 'sets/testing_dict.npy', allow_pickle=True).item()
    valid_dic = np.load(config.main_path + 'sets/validation_dict.npy', allow_pickle=True).item()
    train_dic = np.load(config.main_path + 'sets/training_dict.npy', allow_pickle=True).item()

    train_data = np.load(config.main_path + 'sets/training_list.npy').tolist()
    return train_data, test_dic, valid_dic, user_num, item_num, train_mat, train_dic


class NCFData(data.Dataset):
    def __init__(self, features,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
            else self.features_ps
        labels = self.labels_fill if self.is_training \
            else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label

