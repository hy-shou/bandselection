import pandas as pd
from sklearn.decomposition import FastICA,PCA
from sklearn.model_selection import train_test_split

from baseline_evaluation import *
import h5py

def test_ICA(data_name):
    # indian_pines
    # data = pd.read_csv(r'../data/raw/X_vec.csv', sep=',', header=0, index_col=0)
    # X = data.values[:,:200]
    # y = data.values[:,200]

    # prisma
    # file = h5py.File(r'E:/pyproject/bondselection/data/cm/prisma_20220508_label_v6.h5', 'r')
    # y = file['label'].value
    # X = file['prisma_vec'].value


    f = h5py.File(r'../data/raw/wdc/wdc_for_cls.h5', 'r')
    X = f['test_X'].value
    y = f['test_Y'].value
    # pca = PCA(n_components=20)
    # H = pca.fit_transform(X)  # 基于PCA的成分正交重构信号源

    # band_num = [5,10,15,20,25,30,40,50,60]
    band_num = [5]
    with open(r'../data/ica/' + data_name + '_evaluation_result.txt', 'w') as fw:
        cp = ''
        for num in band_num:
            cp += 'band num:' + str(num) + ':\n'
            # # ICA模型
            ica = FastICA(n_components=num)
            H = ica.fit_transform(X)  # 重构信号
            A_ = ica.mixing_  # 获得估计混合后的矩阵

            # pca = PCA(n_components=20)
            # H = pca.fit_transform(X)  # 基于PCA的成分正交重构信号源

            train_X, test_X, train_Y, test_Y = train_test_split(H, y, test_size=0.9, stratify=y)
            cp += classification_alg(train_X, test_X, train_Y, test_Y)

        fw.write(cp)




if __name__ == "__main__":

    test_ICA('wdc')