from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from evaluation_result import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import json



def KNN_test(train_X,train_Y,test_X,test_Y):

    model = KNeighborsClassifier(n_neighbors=2, weights='distance')
    model.fit(train_X, train_Y)
    score = model.score(test_X, test_Y)
    print('---------------KNN OA',score)
    prd_test_y = model.predict(test_X)
    kappa,acc,aa,report = evaluation(test_Y, prd_test_y)

    return kappa,acc,aa,report


def SVM_test(train_X,train_Y,test_X,test_Y):

    clf = SVC(C=100)
    clf.fit(train_X, train_Y)
    score = clf.score(test_X, test_Y)
    print('---------------SVM OA', score)
    prd_test_y = clf.predict(test_X)
    svc = SVC()
    rfc_s = cross_val_score(svc, train_X, train_Y, cv=10).mean()
    print(rfc_s)
    kappa, acc, aa, report = evaluation(test_Y, prd_test_y)
    return kappa, acc, aa, report


def RF_test(train_X,train_Y,test_X,test_Y):

    rfc = RandomForestClassifier(random_state=20,n_estimators=300)
    rfc.fit(train_X, train_Y)
    score = rfc.score(test_X, test_Y)
    # print('---------------RF OA', score)
    prd_test_y = rfc.predict(test_X)
    kappa, acc, aa, report = evaluation(test_Y, prd_test_y)
    return kappa, acc, aa, report

def Evaluation_RL(bands_filename,matrix_outputname,train_X, train_Y, test_X, test_Y):

    rf_report = ''
    svm_report = ''
    knn_report = ''

    with open(bands_filename) as f:
        # with open(matrix_outputname + '_rf.txt', 'w') as rf_fw:
            with open(matrix_outputname + '_svm.txt', 'w') as svm_fw:
                with open(matrix_outputname + '_knn_rf.txt', 'w') as knn_fw:
                    for line in f:
                        if len(line) < 5 :
                            continue
                        print(line)
                        ids = line[1:len(line) - 2].split(',')
                        bi = [int(i) for i in ids]
                        bi = sorted(bi)

                        testX = test_X[:, bi]
                        trainX = train_X[:, bi]

                        # rf_k, rf_oa, rf_aa, rf_cp = RF_test(trainX, train_Y, testX, test_Y)
                        # rf_report += line + rf_cp

                        svm_k, svm_oa, svm_aa, svm_cp = SVM_test(trainX, train_Y, testX, test_Y)
                        knn_k, knn_oa, knn_aa, knn_cp = KNN_test(trainX, train_Y, testX, test_Y)
                        svm_report += line + svm_cp
                        knn_report += line + knn_cp

                    # rf_fw.write(rf_report)
                    svm_fw.write(svm_report)
                    knn_fw.write(knn_report)



def cross_test(X,y):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc, X, y, cv=10)
    print(rfc_s)
    print(rfc_s.mean())
    plt.plot(range(1, 11), rfc_s, label="Random Forest")

    plt.legend()
    plt.show()