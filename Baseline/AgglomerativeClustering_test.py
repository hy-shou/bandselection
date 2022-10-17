import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,ward,single
from sklearn.metrics.pairwise import cosine_similarity
from baseline_evaluation import *
import h5py
from sklearn.model_selection import train_test_split

def get_cls_label(C):
    #
    n_clusters = C
    # data = pd.read_csv(r'../data/raw/X_vec.csv', sep=',', header=0, index_col=0)
    # X = np.array(data.values[:, :200]).T
    # y = data.values[:, 200]

    # file = h5py.File(r'E:/pyproject/bondselection/data/cm/prisma_20220508_label_v6.h5', 'r')
    # y = file['label'].value
    # X = (file['prisma_vec'].value).T
    #

    f = h5py.File(r'../data/raw/wdc/wdc_for_cls.h5', 'r')
    X = (f['test_X'].value).T

    cls = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(X)
    cls.labels_
    data_c = {}
    for i in range(cls.labels_.shape[0]):
        lable = cls.labels_[i]
        vec = {}
        if lable in data_c.keys():
            vec = data_c[lable]

        vec[i] = X[i]
        data_c[lable] = vec

    # centroid = cls.cluster_centers_

    linkage_matrix = ward(X)
    dendrogram(linkage_matrix)
    # plt.show()
    return data_c
#计算质心

def cal_Cmass(data):
    '''
    input:data(ndarray):数据样本
    output:mass(ndarray):数据样本质心
    '''
    Cmass = np.mean(data,axis=0)
    return Cmass


#计算样本间距离

def distance(x, y, p=2):
    '''
    input:x(ndarray):第一个样本的坐标
          y(ndarray):第二个样本的坐标
          p(int):等于1时为曼哈顿距离，等于2时为欧氏距离
    output:distance(float):x到y的距离
    '''
    dis2 = np.sum(np.abs(x-y)**p) # 计算
    dis = np.power(dis2,1/p)
    return dis

def sorted_list(data,Cmass):
    '''
    input:data(ndarray):数据样本
          Cmass(ndarray):数据样本质心
    output:dis_list(list):排好序的样本到质心距离
    '''
    dis_list = {}
    for k in data.keys():       # 遍历data数据，与质心cmass求距离

        dis_list[k] = distance(Cmass,data[k])
    dis_list = sorted(dis_list.items(),key = lambda x:x[1],reverse = True)      # 排序
    return dis_list[0][0]

def test():
    labels = get_cls_label()
    bid = []
    for k in labels:
        vec = labels[k]
        bcv = []
        for vk in vec:
            bcv.append(vec[vk])
        Cmass = cal_Cmass(bcv)
        id =sorted_list(vec, Cmass)
        bid.append(id)
    print(str(bid))

def get_label(C):
    labels = get_cls_label(C)
    bid = []
    id_index = []
    bs = ''
    for k in labels:
        vec = labels[k]
        bcv = []

        for vk in vec:
            bcv.append(vec[vk].tolist())
            bs += ',' + str(vk)
            id_index.append(vk)

        bcv_mean = pd.DataFrame(bcv).mean().values
        simi = cosine_similarity([bcv_mean], bcv)
        bid.append(id_index[np.argmax(simi)])
        bs +='\n'
        id_index = []
    print(sorted(bid))
    return sorted(bid)

def get_indian_pines_split_data():
    feature = h5py.File(
        '../data/data4classification/indian_pines_randomSampling_0.1_run_1.mat')
    keys = list(feature.keys())
    x_test = feature['x_test'].value.T
    y_test = feature['y_test'].value.T
    Y_test = []
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                Y_test.append(j+1)
                continue

    x_train = feature['x_tra'].value.T
    y_train = feature['y_tra'].value.T
    Y_train = []
    for i in range(y_train.shape[0]):
        for j in range(y_train.shape[1]):
            if y_train[i,j] == 1:
                Y_train.append(j+1)
                continue
    return x_train, Y_train, x_test, Y_test

def get_chongming_data():
    file = h5py.File(r'E:/pyproject/bondselection/data/cm/prisma_label.h5', 'r')
    test_X = file['prisma_test_x']
    test_Y = file['prisma_test_y']
    train_X = file['prisma_train_x']
    train_Y = file['prisma_train_y']
    return train_X, test_X, train_Y, test_Y

def test_Agg():
    with open(r'../data/aggcluster/prisma_bands.txt', 'w') as fw:
        cp = ''
        band_num = [5,10,15,20,25,30,40,50,60]
        vec = ''
        for num in band_num:
            cp += 'band num ' + str(num) + ':\n'
            selectband = get_label(num)
            vec += str(sorted(selectband)) + '\n'

            # # train_X, train_Y,test_X, test_Y = get_indian_pines_split_data()
            # train_X, test_X, train_Y, test_Y = get_chongming_data()
            #
            # test_X = test_X[:, selectband]
            # train_X = train_X[:, selectband]
            #
            # cp += classification_alg(train_X, test_X, train_Y, test_Y)
        fw.write(vec)

if __name__ == "__main__":
    # test()
    # test_Agg()
    get_label(30)