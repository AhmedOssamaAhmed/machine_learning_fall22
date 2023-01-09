import samples
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# load datasets
n_digits_test = 1000
n_digits_training = 5000
n_digits_validation = 1000

n_face_test = 150
n_face_training = 451
n_face_validation = 301


# helper function for data processing
def process_data(data):
    tempdata = np.array(data)
    final_data = []
    for i in range(len(data)):
        final_data.append(tempdata[i].flatten())
    return final_data


# helper function for face data processing
def process_face_data(data):
    new_list = []
    for i in range(len(data)):
        temp_list = []
        for j in range(len(data[i])):
            if len(data[i][j]) != 0:
                temp_list.append(data[i][j])
        new_list.append(temp_list)
    return new_list


# visualization function
def visualize(x_data, labels, title):
    plt.figure()
    plt.subplots_adjust(
        wspace=0.5,
        hspace=0.2)

    plt.suptitle(title)
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_data[i]).axes.get_xaxis().set_visible(False)
        plt.title(labels[i])
    plt.show()


# accuracy change plot
def accuracy_change(data_manhattan, data_euclidean, title):
    x = list(data_manhattan.keys())
    y = list(data_manhattan.values())
    x2 = list(data_euclidean.keys())
    y2 = list(data_euclidean.values())
    plt.plot(x, y, label="manhattan")
    plt.plot(x2, y2, label="euclidean")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------------------------------#

# load digits dataset

# load train digits data
digits_training_data, y_digits_train = samples._test("digits", "digitdata/trainingimages", "digitdata/traininglabels",
                                                     n_digits_training)
x_digits_train = []
for i in range(n_digits_training):
    x_digits_train.append(digits_training_data[i].pixels)

XDigitsTrain = process_data(x_digits_train)

# load test digits data
digits_test_data, y_digits_test = samples._test("digits", "digitdata/testimages", "digitdata/testlabels", n_digits_test)
x_digits_test = []
for i in range(n_digits_test):
    x_digits_test.append(digits_test_data[i].pixels)

XDigitsTest = process_data(x_digits_test)

# load validation digits data
digits_validation_data, y_digits_validation = samples._test("digits", "digitdata/validationimages",
                                                            "digitdata/validationlabels", n_digits_validation)
x_digits_validation = []
for i in range(n_digits_validation):
    x_digits_validation.append(digits_validation_data[i].pixels)

XDigitsValidation = process_data(x_digits_validation)

# ------------------------------------------------------------------------------------------------------#

# load face data set

# load train face data
face_training_data, y_face_train = samples._test("face", "facedata/facedatatrain", "facedata/facedatatrainlabels",
                                                 n_face_training)
x_face_train = []
for i in range(n_face_training):
    x_face_train.append(face_training_data[i].pixels)

faceTrain = process_face_data(x_face_train)
XFaceTrain = process_data(faceTrain)

# load test face data
face_test_data, y_face_test = samples._test("face", "facedata/facedatatest", "facedata/facedatatestlabels", n_face_test)
x_face_test = []
for i in range(n_face_test):
    x_face_test.append(face_test_data[i].pixels)
faceTest = process_face_data(x_face_test)
XFaceTest = process_data(faceTest)

# load validation face data
face_validation_data, y_face_validation = samples._test("face", "facedata/facedatavalidation",
                                                        "facedata/facedatavalidationlabels", n_face_validation)
x_face_validation = []
for i in range(n_face_validation):
    x_face_validation.append(face_validation_data[i].pixels)
faceValid = process_face_data(x_face_validation)
XFaceValidation = process_data(faceValid)


# ------------------------------------------------------------------------------------------------------#

# building different Naïve Bayes Classifier


# fitting the digits dataset to the Gaussian naive bayes classifier
def Gaussian_Naive_Bayes_Digits():
    NB_model_digits = GaussianNB()
    NB_model_digits.fit(XDigitsTrain, y_digits_train)
    predicted_train_digits_naive = NB_model_digits.predict(XDigitsTrain)
    predicted_validation_digits_naive = NB_model_digits.predict(XDigitsValidation)
    predicted_test_digits_naive = NB_model_digits.predict(XDigitsTest)
    print("the accuracy of the digits dataset for the naive bayes classifier is ")
    print(f"the training accuracy is {accuracy_score(y_digits_train, predicted_train_digits_naive) * 100} %")
    print(
        f"the validation accuracy is {accuracy_score(y_digits_validation, predicted_validation_digits_naive) * 100} %")
    print(f"the test accuracy is {accuracy_score(y_digits_test, predicted_test_digits_naive) * 100} %")
    visualize(x_digits_test, predicted_test_digits_naive,
              "samples of digits test dataset with its prediction for naive bayes")
    print("------------------------------------------------------------------------")


# fitting the face dataset to the Gaussian naive classifier
def Gaussian_Naive_Bayes_Face():
    NB_model_face = GaussianNB()
    NB_model_face.fit(XFaceTrain, y_face_train)
    predicted_train_face_naive = NB_model_face.predict(XFaceTrain)
    predicted_validation_face_naive = NB_model_face.predict(XFaceValidation)
    predicted_test_Face_naive = NB_model_face.predict(XFaceTest)
    print("the accuracy of the face dataset for the naive bayes classifier is ")
    print(f"the training accuracy is {accuracy_score(y_face_train, predicted_train_face_naive) * 100} %")
    print(f"the validation accuracy is {accuracy_score(y_face_validation, predicted_validation_face_naive) * 100} %")
    print(f"the test accuracy is {accuracy_score(y_face_test, predicted_test_Face_naive) * 100} %")
    visualize(faceTest, predicted_test_Face_naive, "samples of face test dataset with its prediction for naive bayes")
    print("------------------------------------------------------------------------")


# ------------------------------------------------------------------------------------#

# building the KNN classifier

distance_list = [1, 2]


def KNN_Digits():
    print("training the KNN classifier digit dataset, please wait.... it takes around 4 minutes")
    best_digit_manhattan_k = {}
    best_digit_euclidean_k = {}
    for dist in distance_list:
        k = {}
        for i in range(2, 10):
            KNN_digits = KNeighborsClassifier(n_neighbors=i, p=dist)
            KNN_digits.fit(XDigitsTrain, y_digits_train)
            KNN_digits_validation_accuracy = accuracy_score(y_digits_validation, KNN_digits.predict(XDigitsValidation))
            if dist == distance_list[0]:
                distance_name = "Manhattan"
            else:
                distance_name = "Euclidean"
            print(
                f"the validation accuracy for k = {i} using {distance_name} distance is {KNN_digits_validation_accuracy}")

            k[i] = KNN_digits_validation_accuracy

        if dist == distance_list[0]:
            best_digit_manhattan_k = k
        else:
            if dist == distance_list[1]:
                best_digit_euclidean_k = k

    bestDK = max(best_digit_euclidean_k, key=best_digit_euclidean_k.get)
    accuracy_change(best_digit_manhattan_k, best_digit_euclidean_k, "K vs validation accuracy for digit dataset")

    # using euclidean distance with best k
    print(f"best k for digit data set is {bestDK} using eculidean distance")
    KNN_digits = KNeighborsClassifier(n_neighbors=bestDK)
    KNN_digits.fit(XDigitsTrain, y_digits_train)
    KNN_digits_train_accuracy = accuracy_score(y_digits_train, KNN_digits.predict(XDigitsTrain))
    KNN_digits_validation_accuracy = accuracy_score(y_digits_validation, KNN_digits.predict(XDigitsValidation))
    KNN_digits_test_accuracy = accuracy_score(y_digits_test, KNN_digits.predict(XDigitsTest))
    print(f"the accuracy of the digits dataset for k = {bestDK} for the KNN classifier is ")
    print(f"the training accuracy is {KNN_digits_train_accuracy * 100} %")
    print(f"the validation accuracy is {KNN_digits_validation_accuracy * 100} %")
    print(f"the test accuracy is {KNN_digits_test_accuracy * 100} %")
    visualize(x_digits_test, KNN_digits.predict(XDigitsTest),
              "samples of digits test dataset with its prediction for KNN")
    print("------------------------------------------------------------------------")


# KNN for faces dataset
def KNN_Face():
    print("training the KNN classifier face dataset, please wait.... it takes around 4 minutes")
    best_face_manhattan_k = {}
    best_face_euclidean_k = {}
    for dis in distance_list:
        k = {}
        for i in range(2, 10):
            KNN_face = KNeighborsClassifier(n_neighbors=i, p=dis)
            KNN_face.fit(XFaceTrain, y_face_train)
            KNN_face_validation_accuracy = accuracy_score(y_face_validation, KNN_face.predict(XFaceValidation))
            if dis == distance_list[0]:
                distance_name = "Manhattan"
            else:
                distance_name = "Euclidean"
            print(
                f"the validation accuracy for k = {i} using {distance_name} distance is {KNN_face_validation_accuracy}")
            k[i] = KNN_face_validation_accuracy
        if dis == distance_list[0]:
            best_face_manhattan_k = k
        else:
            if dis == distance_list[1]:
                best_face_euclidean_k = k

    bestFK = max(best_face_euclidean_k, key=best_face_euclidean_k.get)
    accuracy_change(best_face_manhattan_k, best_face_euclidean_k, "K vs validation accuracy for face dataset")
    print(f"best k for face data set is {bestFK} using eculidean distance")
    KNN_face = KNeighborsClassifier(n_neighbors=bestFK)
    KNN_face.fit(XFaceTrain, y_face_train)
    KNN_face_train_accuracy = accuracy_score(y_face_train, KNN_face.predict(XFaceTrain))
    KNN_face_validation_accuracy = accuracy_score(y_face_validation, KNN_face.predict(XFaceValidation))
    KNN_face_test_accuracy = accuracy_score(y_face_test, KNN_face.predict(XFaceTest))
    print(f"the accuracy of the face dataset for k = {bestFK} for the KNN classifier is ")
    print(f"the training accuracy is {KNN_face_train_accuracy * 100} %")
    print(f"the validation accuracy is {KNN_face_validation_accuracy * 100} %")
    print(f"the test accuracy is {KNN_face_test_accuracy * 100} %")
    visualize(faceTest, KNN_face.predict(XFaceTest), "samples of face test dataset with its prediction for KNN")
    print("------------------------------------------------------------------------")


# Decision Tree for Digits dataset
def DecisionTree_Digits():
    print("Tuning the hyperparameters...... please wait")
    for n in range(15, 20):
        min_samples_dict = {}
        for i in range(2, 6):
            DecisionTree_model = DecisionTreeClassifier(max_depth=n, min_samples_split=i, random_state=0)
            DecisionTree_model.fit(XDigitsTrain, y_digits_train)
            DecisionTree_digits_validation_accuracy = accuracy_score(y_digits_validation,
                                                                     DecisionTree_model.predict(XDigitsValidation))
            min_samples_dict[i] = DecisionTree_digits_validation_accuracy
            x = list(min_samples_dict.keys())
            y = list(min_samples_dict.values())
            plt.plot(x, y, label=f"max depth = {n}")
    plt.title("max depth vs min sample split")
    plt.xlabel("min samples split")
    plt.ylabel("validation accuracy")
    plt.legend()
    plt.show()

    DecisionTree_model = DecisionTreeClassifier(min_samples_split=3, max_depth=18, random_state=0)
    DecisionTree_model.fit(XDigitsTrain, y_digits_train)
    predicted_train_digit_decision_tree = DecisionTree_model.predict(XDigitsTrain)
    predicted_validation_digit_decision_tree = DecisionTree_model.predict(XDigitsValidation)
    predicted_test_digit_decision_tree = DecisionTree_model.predict(XDigitsTest)
    print("the accuracy of the digits dataset for the decision tree is ")
    print(f"the training accuracy is {accuracy_score(y_digits_train, predicted_train_digit_decision_tree) * 100} %")
    print(
        f"the validation accuracy is {accuracy_score(y_digits_validation, predicted_validation_digit_decision_tree) * 100} %")
    print(f"the test accuracy is {accuracy_score(y_digits_test, predicted_test_digit_decision_tree) * 100} %")
    print("------------------------------------------------------------------------")
    visualize(x_digits_test, predicted_test_digit_decision_tree,
              "samples of digits test dataset with its prediction for Decision Tree")


# Decision Tree for Faces dataset
def DecisionTree_Faces():
    print("Tuning the hyperparameters...... please wait")
    for n in range(8, 15):
        min_samples_dict = {}
        for i in range(2, 6):
            DecisionTree_model = DecisionTreeClassifier(max_depth=n, min_samples_split=i, random_state=0)
            DecisionTree_model.fit(XFaceTrain, y_face_train)
            DecisionTree_face_validation_accuracy = accuracy_score(y_face_validation,
                                                                   DecisionTree_model.predict(XFaceValidation))
            min_samples_dict[i] = DecisionTree_face_validation_accuracy
            x = list(min_samples_dict.keys())
            y = list(min_samples_dict.values())
            plt.plot(x, y, label=f"max depth = {n}")
    plt.title("max depth vs min sample split")
    plt.xlabel("min samples split")
    plt.ylabel("validation accuracy")
    plt.legend()
    plt.show()

    DecisionTree_model = DecisionTreeClassifier(min_samples_split=2, max_depth=12, random_state=0)
    DecisionTree_model.fit(XFaceTrain, y_face_train)
    predicted_train_face_decision_tree = DecisionTree_model.predict(XFaceTrain)
    predicted_validation_face_decision_tree = DecisionTree_model.predict(XFaceValidation)
    predicted_test_face_decision_tree = DecisionTree_model.predict(XFaceTest)
    print("the accuracy of the face dataset for the decision tree is ")
    print(f"the training accuracy is {accuracy_score(y_face_train, predicted_train_face_decision_tree) * 100} %")
    print(
        f"the validation accuracy is {accuracy_score(y_face_validation, predicted_validation_face_decision_tree) * 100} %")
    print(f"the test accuracy is {accuracy_score(y_face_test, predicted_test_face_decision_tree) * 100} %")
    print("------------------------------------------------------------------------")
    visualize(faceTest, predicted_test_face_decision_tree,
              "samples of faces test dataset with its prediction for Decision Tree")


# MLP Digits
def MLP_Digits():
    # params = {
    #     'activation': ['relu', 'identity', 'logistic', 'tanh'],
    #     'solver': ['lbfgs', 'sgd', 'adam'],
    #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
    # }
    # grid_search = GridSearchCV(
    #     MLPClassifier(verbose=True),
    #     param_grid=params,
    #     return_train_score=True,
    #     verbose=True,
    #     n_jobs=-1
    # )
    # grid_search.fit(XDigitsTrain, y_digits_train)
    # print(grid_search.best_params_)

    # the best params are {'activation': 'relu', 'learning_rate': 'adaptive', 'solver': 'adam'}
    # now lets try changing the hidden_layer_sizes and display the accuracy change using the above parameters

    # accuracy = {}
    # for i in range(50,1000, 50):
    #     mlp_digits = MLPClassifier(
    #         hidden_layer_sizes=(i,),
    #         activation='relu',
    #         solver='adam',
    #         learning_rate='adaptive',
    #         verbose=False)
    #     print(i)
    #     mlp_digits.fit(XDigitsTrain,y_digits_train)
    #     mlp_digits_training_accuracy = accuracy_score(y_digits_train,mlp_digits.predict(XDigitsTrain))
    #     print(f"training accuracy for MPL for digits dataset is {mlp_digits_training_accuracy*100}%")
    #     mlp_digits_validation_accuracy = accuracy_score(y_digits_validation,mlp_digits.predict(XDigitsValidation))
    #     print(f"validation accuracy for MPL for digits dataset is {mlp_digits_validation_accuracy*100}%")
    #     mlp_digits_test_accuracy = accuracy_score(y_digits_test,mlp_digits.predict(XDigitsTest))
    #     print(f"training accuracy for MPL for digits dataset is {mlp_digits_test_accuracy*100}%")
    #     accuracy[i] = mlp_digits_validation_accuracy
    #     x = list(accuracy.keys())
    #     y = list(accuracy.values())
    #     plt.plot(x, y, label=f"layers = {i}")
    # plt.legend()
    # plt.show()

    # so we can see the best hidden layers sizes is 350

    #lets finally tune the alpha and the initial
    # alpha = [0.0001,0.001,0.01,0.1]
    # learning_rate_init =[0.0001,0.001,0.01,0.1]
    # for i in alpha:
    #     accuracy = {}
    #     for j in learning_rate_init:
    #         mlp_digits = MLPClassifier(
    #             hidden_layer_sizes=(350,),
    #             activation='relu',
    #             solver='adam',
    #             learning_rate='adaptive',
    #             alpha=i,
    #             learning_rate_init=j,
    #             verbose=True)
    #         print(i,j)
    #         mlp_digits.fit(XDigitsTrain, y_digits_train)
    #         mlp_digits_training_accuracy = accuracy_score(y_digits_train, mlp_digits.predict(XDigitsTrain))
    #         print(f"training accuracy for MPL for digits dataset is {mlp_digits_training_accuracy * 100}%")
    #         mlp_digits_validation_accuracy = accuracy_score(y_digits_validation,
    #                                                         mlp_digits.predict(XDigitsValidation))
    #         print(f"validation accuracy for MPL for digits dataset is {mlp_digits_validation_accuracy * 100}%")
    #         mlp_digits_test_accuracy = accuracy_score(y_digits_test, mlp_digits.predict(XDigitsTest))
    #         print(f"training accuracy for MPL for digits dataset is {mlp_digits_test_accuracy * 100}%")
    #         accuracy[j] = mlp_digits_validation_accuracy
    #     x = list(accuracy.keys())
    #     y = list(accuracy.values())
    #     plt.plot(x, y, label=f"alpha {i} ")
    # plt.legend()
    # plt.show()

    # the best alpha is 0.001 with learning init of 0.01

    # so the final model is as the following
    mlp_digits = MLPClassifier(
                hidden_layer_sizes=(350,),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.01,
                alpha=0.001,
                verbose=False)
    mlp_digits.fit(XDigitsTrain,y_digits_train)
    mlp_digits_training_accuracy = accuracy_score(y_digits_train,mlp_digits.predict(XDigitsTrain))
    print("the accuracy of the digits dataset for the MLP is ")
    print(f"training accuracy for MLP for digits dataset is {mlp_digits_training_accuracy*100}%")
    mlp_digits_validation_accuracy = accuracy_score(y_digits_validation,mlp_digits.predict(XDigitsValidation))
    print(f"validation accuracy for MLP for digits dataset is {mlp_digits_validation_accuracy*100}%")
    mlp_digits_test_accuracy = accuracy_score(y_digits_test,mlp_digits.predict(XDigitsTest))
    print(f"training accuracy for MLP for digits dataset is {mlp_digits_test_accuracy*100}%")
    predicted =mlp_digits.predict(XDigitsTest)
    visualize(x_digits_test, predicted,
              "samples of digits test dataset with its prediction for MLP")
    print("------------------------------------------------------------------------")

# MLP Faces
def MLP_Faces():
    # params = {
    #     'activation': ['relu', 'identity', 'logistic', 'tanh'],
    #     'solver': ['lbfgs', 'sgd', 'adam'],
    #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
    # }
    # grid_search = GridSearchCV(
    #     MLPClassifier(verbose=True),
    #     param_grid=params,
    #     return_train_score=True,
    #     verbose=True,
    #     n_jobs=-1
    # )
    # grid_search.fit(XFaceTrain, y_face_train)
    # print(grid_search.best_params_)

    # the best params are {'activation': 'relu', 'learning_rate': 'constant', 'solver': 'adam'}
    # now lets try changing the hidden_layer_sizes and display the accuracy change using the above parameters

    # accuracy = {}
    # for i in range(500,1500, 50):
    #     mlp_face = MLPClassifier(
    #         hidden_layer_sizes=(i,),
    #         activation='relu',
    #         solver='adam',
    #         learning_rate='constant',
    #         verbose=False)
    #     print(i)
    #     mlp_face.fit(XFaceTrain,y_face_train)
    #     mlp_face_training_accuracy = accuracy_score(y_face_train,mlp_face.predict(XFaceTrain))
    #     print(f"training accuracy for MPL for face dataset is {mlp_face_training_accuracy*100}%")
    #     mlp_face_validation_accuracy = accuracy_score(y_face_validation,mlp_face.predict(XFaceValidation))
    #     print(f"validation accuracy for MPL for face dataset is {mlp_face_validation_accuracy*100}%")
    #     mlp_face_test_accuracy = accuracy_score(y_face_test,mlp_face.predict(XFaceTest))
    #     print(f"training accuracy for MPL for face dataset is {mlp_face_test_accuracy*100}%")
    #     accuracy[i] = mlp_face_validation_accuracy
    #     x = list(accuracy.keys())
    #     y = list(accuracy.values())
    #     plt.plot(x, y, label=f"layers = {i}")
    # plt.show()

    # so we can see the best hidden layers sizes is 600

    # lets finally tune the alpha and the initial
    # alpha = [0.0001,0.001,0.01,0.1]
    # learning_rate_init =[0.0001,0.001,0.01,0.1]
    # for i in alpha:
    #     accuracy = {}
    #     for j in learning_rate_init:
    #         mlp_face = MLPClassifier(
    #             hidden_layer_sizes=(600,),
    #             activation='relu',
    #             solver='adam',
    #             learning_rate='adaptive',
    #             alpha=i,
    #             learning_rate_init=j,
    #             verbose=True)
    #         print(i,j)
    #         mlp_face.fit(XFaceTrain, y_face_train)
    #         mlp_face_training_accuracy = accuracy_score(y_face_train, mlp_face.predict(XFaceTrain))
    #         print(f"training accuracy for MPL for face dataset is {mlp_face_training_accuracy * 100}%")
    #         mlp_face_validation_accuracy = accuracy_score(y_face_validation,
    #                                                         mlp_face.predict(XFaceValidation))
    #         print(f"validation accuracy for MPL for face dataset is {mlp_face_validation_accuracy * 100}%")
    #         mlp_face_test_accuracy = accuracy_score(y_face_test, mlp_face.predict(XFaceTest))
    #         print(f"training accuracy for MPL for face dataset is {mlp_face_test_accuracy * 100}%")
    #         accuracy[j] = mlp_face_validation_accuracy
    #     x = list(accuracy.keys())
    #     y = list(accuracy.values())
    #     plt.plot(x, y, label=f"alpha {i} ")
    # plt.legend()
    # plt.show()

    # the best alpha is 0.0001 with learning init of 0.001

    # so the final model is as the following
    mlp_face = MLPClassifier(
                hidden_layer_sizes=(600,),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                alpha=0.0001,
                verbose=False)
    mlp_face.fit(XFaceTrain,y_face_train)
    mlp_face_training_accuracy = accuracy_score(y_face_train,mlp_face.predict(XFaceTrain))
    print("the accuracy of the faces dataset for the MLP is ")
    print(f"training accuracy for MLP for face dataset is {mlp_face_training_accuracy*100}%")
    mlp_face_validation_accuracy = accuracy_score(y_face_validation,mlp_face.predict(XFaceValidation))
    print(f"validation accuracy for MLP for face dataset is {mlp_face_validation_accuracy*100}%")
    mlp_face_test_accuracy = accuracy_score(y_face_test,mlp_face.predict(XFaceTest))
    print(f"training accuracy for MLP for face dataset is {mlp_face_test_accuracy*100}%")
    predicted =mlp_face.predict(XFaceTest)
    visualize(faceTest, predicted,
              "samples of face test dataset with its prediction for MLP")
    print("------------------------------------------------------------------------")


kernel = ['linear', 'poly', 'rbf']
gamma = ['auto', 'scale']

# SVM for Digits dataset
def SVM_Digits():
    print("Tuning the hyperparameters...... please wait")
    for k in kernel:
        params = {}
        for g in gamma:
            SVM_model = SVC(kernel=k, gamma=g, degree=2)
            SVM_model.fit(XDigitsTrain, y_digits_train)
            SVM_digits_validation_accuracy = accuracy_score(y_digits_validation, SVM_model.predict(XDigitsValidation))
            params[g] = SVM_digits_validation_accuracy
            x = list(params.keys())
            y = list(params.values())
            plt.plot(x, y, label=f"Kernel = {k}")
    plt.title("kernel vs gamma")
    plt.xlabel("gamma")
    plt.ylabel("validation accuracy")
    plt.legend()
    plt.show()

    # detect the polynomial degree value that will give the highest accuracy
    poly = {}
    for degree in range(1,5):
        SVM_model = SVC(kernel='poly', gamma='scale', degree=degree)
        SVM_model.fit(XDigitsTrain, y_digits_train)
        Poly_digits_validation_accuracy = accuracy_score(y_digits_validation, SVM_model.predict(XDigitsValidation))
        poly[degree] = Poly_digits_validation_accuracy
        x = list(poly.keys())
        y = list(poly.values())
        plt.plot(x, y)
    plt.title("different degrees of polynomial")
    plt.xlabel("degree of polynomial")
    plt.ylabel("validation accuracy")
    plt.show()

    SVM_model = SVC(kernel='poly', gamma='scale', degree=2)
    SVM_model.fit(XDigitsTrain, y_digits_train)
    predicted_train_digits_SVM = SVM_model.predict(XDigitsTrain)
    predicted_validation_digits_SVM = SVM_model.predict(XDigitsValidation)
    predicted_test_digits_SVM = SVM_model.predict(XDigitsTest)
    print("the accuracy of the digits dataset for the SVM classifier is ")
    print(f"the training accuracy is {accuracy_score(y_digits_train, predicted_train_digits_SVM) * 100} %")
    print(f"the validation accuracy is {accuracy_score(y_digits_validation, predicted_validation_digits_SVM) * 100} %")
    print(f"the test accuracy is {accuracy_score(y_digits_test, predicted_test_digits_SVM) * 100} %")
    print("------------------------------------------------------------------------")
    visualize(x_digits_test, predicted_test_digits_SVM,
              "samples of digits test dataset with its prediction for SVM")

# SVM for Faces dataset
def SVM_Faces():
    print("Tuning the hyperparameters...... please wait")
    for k in kernel:
        params = {}
        for g in gamma:
            SVM_model = SVC(kernel=k, gamma=g, degree=2)
            SVM_model.fit(XFaceTrain, y_face_train)
            SVM_faces_validation_accuracy = accuracy_score(y_face_validation, SVM_model.predict(XFaceValidation))
            params[g] = SVM_faces_validation_accuracy
            x = list(params.keys())
            y = list(params.values())
            plt.plot(x, y, label=f"Kernel = {k}")
    plt.title("kernel vs gamma")
    plt.xlabel("gamma")
    plt.ylabel("validation accuracy")
    plt.legend()
    plt.show()

    # detect the polynomial degree value that will give the highest accuracy
    poly = {}
    for degree in range(1,5):
        SVM_model = SVC(kernel='poly', gamma='scale', degree=degree)
        SVM_model.fit(XFaceTrain, y_face_train)
        Poly_faces_validation_accuracy = accuracy_score(y_face_validation, SVM_model.predict(XFaceValidation))
        poly[degree] = Poly_faces_validation_accuracy
        x = list(poly.keys())
        y = list(poly.values())
        plt.plot(x, y)
    plt.title("different degrees of polynomial")
    plt.xlabel("degree of polynomial")
    plt.ylabel("validation accuracy")
    plt.show()

    SVM_model = SVC(kernel='poly', gamma='scale', degree=2)
    SVM_model.fit(XFaceTrain, y_face_train)
    predicted_train_faces_SVM = SVM_model.predict(XFaceTrain)
    predicted_validation_faces_SVM = SVM_model.predict(XFaceValidation)
    predicted_test_faces_SVM = SVM_model.predict(XFaceTest)
    print("the accuracy of the faces dataset for the SVM classifier is ")
    print(f"the training accuracy is {accuracy_score(y_face_train, predicted_train_faces_SVM) * 100} %")
    print(f"the validation accuracy is {accuracy_score(y_face_validation, predicted_validation_faces_SVM) * 100} %")
    print(f"the test accuracy is {accuracy_score(y_face_test, predicted_test_faces_SVM) * 100} %")
    print("------------------------------------------------------------------------")
    visualize(faceTest, predicted_test_faces_SVM,
              "samples of faces test dataset with its prediction for svm")


if __name__ == "__main__":
    # Gaussian_Naive_Bayes_Digits()
    # Gaussian_Naive_Bayes_Face()
    # KNN_Digits()
    # KNN_Face()
    DecisionTree_Digits()
    DecisionTree_Faces()
    MLP_Digits()
    MLP_Faces()
    SVM_Digits()
    SVM_Faces()
