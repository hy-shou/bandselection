import os
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_evaluation import *


def test_PCA(file_name):
    with open(r'../data/' + file_name + '/evaluation_result.txt', 'w') as fw:
        names = os.listdir('../data/' + file_name + '/vec/')
        cp = ''
        for name in names:
            print(name)
            cp += name + '\n'
            H_data = pd.read_csv('../data/' + file_name + '/vec/' + name, sep=',', header=0, index_col=0)
            data = H_data.values
            X = H_data.values[:, :data.shape[1]-1]
            y = H_data.values[:, data.shape[1]-1]
            train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.9, stratify=y)
            cp += classification_alg(train_X, test_X, train_Y, test_Y)
        fw.write(cp)

if __name__ == "__main__":

    test_PCA('mvpca/prisma')
    test_PCA('pca/prisma')