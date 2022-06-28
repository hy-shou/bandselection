from load_data import *
from sk_classification import *
from evaluation_result import *
from sklearn.model_selection import train_test_split
from load_data import *
from sk_classification import *
data_path = r'../data/'
band_num = 'indian_pines/60'

def cm_data_rf_evaluation():

    file = h5py.File(r'E:/pyproject/bondselection/data/cm/prisma_20220508_label_v6.h5', 'r')
    label = file['label'].value
    prisma_vec = file['prisma_vec'].value
    s2_vec = file['s2_vec'].value
    print(len(label))

    # train_X, test_X, train_Y, test_Y = train_test_split(prisma_vec, label, test_size=0.9, stratify=label)

    file = h5py.File(r'E:/pyproject/bondselection/data/cm/prisma_label.h5', 'r')
    test_X = file['prisma_test_x']
    test_Y = file['prisma_test_y']
    train_X = file['prisma_train_x']
    train_Y = file['prisma_train_y']


    bands_id = [4, 8, 12, 14, 16, 20, 22, 25, 31, 34, 36, 46, 49, 51, 53, 58, 60, 61, 73, 76, 78, 79, 80, 83, 89, 92, 105, 106, 107, 116, 134, 138,  148, 172, 177, 181, 185,  203, 207, 233]

    # [3, 6, 8, 14, 34, 37, 39, 45, 50, 51, 52, 53, 55, 57, 77, 82, 85, 98, 106, 108, 126, 137, 144, 149, 155, 156, 158, 163, 164, 165, 172, 176, 177, 182, 184, 185, 193, 194, 195, 199]
    #
    test_X = test_X[:, bands_id]
    train_X = train_X[:, bands_id]

    KNN_test(train_X,train_Y,test_X,test_Y)
    SVM_test(train_X, train_Y, test_X, test_Y)
    RF_test(train_X, train_Y, test_X, test_Y)






def indian_pines_rf_evaluation():
    train_X, train_Y, test_X, test_Y  = get_indian_pines_split_data()
    bands_filename = data_path + 'ddqn/' + band_num + '_bands.txt'
    matrix_outputname = data_path + 'ddqn/' + band_num + '_matrix_'
    Evaluation_RL(bands_filename, matrix_outputname, train_X, train_Y, test_X, test_Y)


def indian_pines_sb_evaluation(selected_bands):
    train_X, train_Y, test_X, test_Y = get_indian_pines_split_data()
    testX = test_X[:, selected_bands]
    trainX = train_X[:, selected_bands]
    # RF_test(trainX,train_Y,testX,test_Y)
    SVM_test(trainX, train_Y, testX, test_Y)
    KNN_test(trainX, train_Y, testX, test_Y)

if __name__ == "__main__":
    # RL test
    # indian_pines_rf_evaluation()
    # selected_bands = [4, 5, 6, 8, 9, 11, 14, 18, 19, 21, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 44, 49, 50, 54, 55, 62, 65, 66, 69, 97, 106, 107, 109, 111, 112, 114, 119, 120, 125, 126, 127, 130, 132, 133, 134, 148, 149, 156, 157, 167, 179, 181, 185, 186, 188, 191]

    # indian_pines_sb_evaluation(selected_bands)

    cm_data_rf_evaluation()

