import numpy as np

def treatment2index(treat, treat_clip_num):
    # the discretizing function
    clip_size = 1 / treat_clip_num
    return int(treat / clip_size + 0.499)  # avoid edge value


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# notice: if the index start from 0, the code should be changed
def DP_solve(C, K, h, T0, eps):
    """
    calculate the best ratios for a user given ADRF matrix
    parameters:
        C: category number(1,2,...,C)
        K: number of discrete treatment intervals(0,1,2,...,K)
        h: h[i-1][j] represents the average reward on category i at the cost of the treatment in jth interval
        T0: the original treatment proportion T0[i] represents the proportion of ith category
        eps: the allowed deviation of the proposed treatment
    output:
    f[][]: best reward
    T[1:]: best eposure rate
    """
    f = np.zeros([C + 1, K + 1])
    best_T = np.zeros([C + 1, K + 1])  # best_T[i][j] saves the best treatment of category i in f[i][j]
    # f[i][j] represents the maximum reward using the \\
    # first (i+1) categories at the cost of the treatment in kth interval
    for i in range(1, C + 1):
        for j in range(1, K + 1):
            f[i][j] = f[i - 1][j]
            for k in range(1, j + 1):
                if (k < T0[i - 1] - eps):
                    continue
                if (k > T0[i - 1] + eps):
                    continue
                if f[i - 1][j - k] + k * h[i - 1][k] >= f[i][j]:  # to add more new categories
                    f[i][j] = f[i - 1][j - k] + k * h[i - 1][k]
                    best_T[i][j] = k
                # f[i][j] = max(f[i][j], f[i-1][j-k] + h[i][k])
    i = C
    j = K
    T = [0] * (C + 1)
    while ((i > 0) & (j > 0)):
        T[i] += best_T[i][j] / K
        j -= int(best_T[i][j])
        i -= 1
    return f[C][K], T[1:]


def rank_second(ele):
    return ele[1]


def calc_T(t_seq, item_category, cate_num):
    '''
    :param t_seq: an item sequence
    :param item_category: dict saving each item's category
    :param cate_num: number of categories
    :return: the category distribution
    '''
    cate_dis = [0] * cate_num
    for item in t_seq:
        for c in item_category[item]:
            cate_dis[c] += 1 / len(item_category[item])
    cate_dis = [c / len(t_seq) for c in cate_dis]
    return cate_dis


def makeset(a):
    '''
    remove repeated items in a list
    :param a: a list
    :return: a clipped list
    '''
    b = a[::-1]
    for i in a:
        if a.count(i) > 1:
            for j in range(0, b.count(i) - 1):
                b.remove(i)
    return b[::-1]