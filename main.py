import samples
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric

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
        # plt.axes.get_yaxis().set_visible(False)
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
            print(f"the validation accuracy for k = {i} using {distance_name} distance")

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
            print(f"the validation accuracy for k = {i} using {distance_name} distance")
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

# for training use :
# XDigitsTrain, y_digits_train
# XFaceTrain, y_face_train

# for testing use :
# XDigitsTest,y_digits_test
# XFaceTest, y_face_test

# for validation use :
# XDigitsValidation, y_digits_validation
# XFaceValidation, y_face_validation

# start SVM or Decision tree or MLP here ....

if __name__ == "__main__":
    Gaussian_Naive_Bayes_Digits()
    Gaussian_Naive_Bayes_Face()
    KNN_Digits()
    KNN_Face()