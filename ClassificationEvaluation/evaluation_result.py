import sklearn.metrics as sm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

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

    # report = cp + '\n-----------------------------\n kappa: ' + str(acc) + '\n' +str(aa) + '\n' + str(kappa) + '\n'
    report = '\n-----------------------------\n kappa: ' + str(acc) + '\n' + str(aa) + '\n' + str(kappa) + '\n'
    # print('宏平均精确率:', sm.precision_score(test_Y, prd_test_y, average='macro'))  # 预测宏平均精确率输出
    # print('微平均精确率:', sm.precision_score(test_Y, prd_test_y, average='micro'))  # 预测微平均精确率输出
    # print('加权平均精确率:', sm.precision_score(test_Y, prd_test_y, average='weighted'))  # 预测加权平均精确率输出
    print(report)
    return kappa,acc,aa,report
