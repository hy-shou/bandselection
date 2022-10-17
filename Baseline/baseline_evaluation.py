from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sm
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import pandas as pd

def classification_alg(train_X, test_X, train_Y, test_Y):
    cp = ''
    model = KNeighborsClassifier(n_neighbors=2, weights='distance')
    model.fit(train_X, train_Y)
    score = model.score(test_X, test_Y)
    print('---------------KNN OA', score)
    prd_test_y = model.predict(test_X)
    cp += 'KNN:\n'
    knn_k, knn_oa, knn_aa, knn_cp = evaluation(test_Y, prd_test_y)
    cp += knn_cp

    clf = SVC(C=50, kernel='rbf')
    clf.fit(train_X, train_Y)
    score = clf.score(test_X, test_Y)
    print('---------------SVM OA', score)
    prd_test_y = clf.predict(test_X)
    svm_k, svm_oa, svm_aa, svm_cp = evaluation(test_Y, prd_test_y)
    cp += '\nSVM:\n'
    cp += svm_cp

    rfc = RandomForestClassifier(random_state=20, n_estimators=300)
    rfc.fit(train_X, train_Y)
    score = rfc.score(test_X, test_Y)
    print('---------------RF OA', score)
    prd_test_y = rfc.predict(test_X)
    rf_k, rf_oa, rf_aa, rf_cp = evaluation(test_Y, prd_test_y)
    cp += '\nRF:\n'
    cp += rf_cp
    return cp


def evaluation(test_Y, prd_test_y):

    cm = sm.confusion_matrix(test_Y, prd_test_y)
    # print("---------------混淆矩阵\n", cm)
    dataframe = pd.DataFrame(cm)
    # dataframe.to_csv('./results/' + cl + '_matrix.csv', index=False, sep='\t')

    cp = sm.classification_report(test_Y, prd_test_y)
    # print("---------------分类报告\n", cp)

    kappa = cohen_kappa_score(test_Y, prd_test_y)
    # print("---------------kappa\n", kappa)

    acc = accuracy_score(test_Y, prd_test_y)
    aa = sm.precision_score(test_Y, prd_test_y, average='macro')

    # print("---------------acc\n",accuracy_score(test_Y, prd_test_y))

    report = cp + '\n-----------------------------\n kappa: ' + str(acc) + '\n' +str(aa) + '\n' + str(kappa) + '\n'
    report = '\n-----------------------------\n kappa: ' + str(acc) + '\n' + str(aa) + '\n' + str(kappa) + '\n'
    # print('宏平均精确率:', sm.precision_score(test_Y, prd_test_y, average='macro'))  # 预测宏平均精确率输出
    # print('微平均精确率:', sm.precision_score(test_Y, prd_test_y, average='micro'))  # 预测微平均精确率输出
    # print('加权平均精确率:', sm.precision_score(test_Y, prd_test_y, average='weighted'))  # 预测加权平均精确率输出
    print(report)
    return kappa,acc,aa,report

