import pandas as pd
import numpy as np
import scipy.io as sio
import h5py
from sklearn.feature_selection import mutual_info_classif
data_path = r'../data/'


def get_pca_data(filename):
    data = pd.read_csv(filename, sep=',',header=0,index_col=0)
    return data

def get_indian_pines_split_data():
    feature = h5py.File(
        data_path+'data4classification/indian_pines_randomSampling_0.1_run_1.mat')
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

def get_indian_pines_all_data():
    train_X, train_Y, test_X, test_Y = get_indian_pines_split_data()
    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_Y, test_Y), axis=0)
    return X,y

def get_drl_nb_data(selected_bands):
    train_X, train_Y, test_X, test_Y = get_indian_pines_split_data()
    # data = sio.loadmat(data_path+ 'DRL4BS_result/drl_30_bands_indian_pines.mat')
    # selected_b = data['selected_bands']
    bands_id = [int(bid) for bid in selected_bands[0]]

    test_X = test_X[:,bands_id]
    train_X = train_X[:,bands_id]

    return train_X, train_Y, test_X, test_Y

def get_chongming_ig_ig():
    feature_names = list(range(239))

    file = h5py.File(r'E:/pyproject/bondselection/data/cm/prisma_20220508_label_v6.h5', 'r')
    label = file['label'].value
    prisma_vec = file['prisma_vec'].value
    s2_vec = file['s2_vec'].value
    res = dict(zip(feature_names,
                   mutual_info_classif(prisma_vec, label, discrete_features=True)
                   ))
    print(res)
    df = pd.DataFrame([res])
    df.to_csv(r'../data/cm/prisma_ig.csv')

    fw = open(r'../data/cm/information_divergence.txt','w')


    vec = []
    d_order = sorted(res.items(), key=lambda x: x[1], reverse=True)
    for v in d_order:
        fw.write(str(v[0]))
        fw.write('\t')
        fw.write(str(v[1]))
        fw.write('\n')

    fw.close()

def get_chongming_data():

    file = h5py.File(r'E:/pyproject/bondselection/data/cm/prisma_20220508_label_v6.h5', 'r')
    label = file['label'].value
    prisma_vec = file['prisma_vec'].value
    s2_vec = file['s2_vec'].value

    df = pd.DataFrame(prisma_vec)
    df['label'] = label
    df.to_csv(r'../data/cm/chongming_vec.csv')
if __name__ == "__main__":
    get_chongming_ig_ig()

