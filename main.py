import samples
import numpy as np
from matplotlib import pyplot as plt

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
def visualize(x_data, labels,title):
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
def accuracy_change(data,title):
    x = list(data.keys())
    y = list(data.values())
    plt.plot(x, y)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.show()


# ------------------------------------------------------------------------------------------------------#

# load digits dataset

# load train digits data
digits_training_data, y_digits_train = samples._test("digits", n_digits_training)
x_digits_train = []
for i in range(n_digits_training):
    x_digits_train.append(digits_training_data[i].pixels)

XDigitsTrain = process_data(x_digits_train)

# load test digits data
digits_test_data, y_digits_test = samples._test("digits", n_digits_test)
x_digits_test = []
for i in range(n_digits_test):
    x_digits_test.append(digits_test_data[i].pixels)

XDigitsTest = process_data(x_digits_test)

# load validation digits data
digits_validation_data, y_digits_validation = samples._test("digits", n_digits_validation)
x_digits_validation = []
for i in range(n_digits_validation):
    x_digits_validation.append(digits_validation_data[i].pixels)

XDigitsValidation = process_data(x_digits_validation)

# ------------------------------------------------------------------------------------------------------#

# load face data set

# load train face data
face_training_data, y_face_train = samples._test("face", n_face_training)
x_face_train = []
for i in range(n_face_training):
    x_face_train.append(face_training_data[i].pixels)
faceTrain = process_face_data(x_face_train)
XFaceTrain = process_data(faceTrain)

# load test face data
face_test_data, y_face_test = samples._test("face", n_face_test)
x_face_test = []
for i in range(n_face_test):
    x_face_test.append(face_test_data[i].pixels)
faceTest = process_face_data(x_face_test)
XFaceTest = process_data(faceTest)

# load validation face data
face_validation_data, y_face_validation = samples._test("face", n_face_validation)
x_face_validation = []
for i in range(n_face_validation):
    x_face_validation.append(face_validation_data[i].pixels)
faceValid = process_face_data(x_face_validation)
XFaceValidation = process_data(faceValid)

# ------------------------------------------------------------------------------------------------------#

# building Naïve Bayes Classifier

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# fitting the digits dataset to the naive bayes classifier
NB_model_digits = GaussianNB()
NB_model_digits.fit(XDigitsTrain, y_digits_train)
predicted_train_digits_naive = NB_model_digits.predict(XDigitsTrain)
predicted_validation_digits_naive = NB_model_digits.predict(XDigitsValidation)
predicted_test_digits_naive = NB_model_digits.predict(XDigitsTest)
visualize(x_digits_test, predicted_test_digits_naive,"samples of digits test dataset with its prediction for naive bayes")
print("the accuracy of the digits dataset for the naive bayes classifier is ")
print(f"the training accuracy is {accuracy_score(y_digits_train, predicted_train_digits_naive) * 100} %")
print(f"the validation accuracy is {accuracy_score(y_digits_validation, predicted_validation_digits_naive) * 100} %")
print(f"the test accuracy is {accuracy_score(y_digits_test, predicted_test_digits_naive) * 100} %")
print("------------------------------------------------------------------------")

# fitting the face dataset to the naive classifier
NB_model_face = GaussianNB()
NB_model_face.fit(XFaceTrain, y_face_train)
predicted_train_face_naive = NB_model_face.predict(XFaceTrain)
predicted_validation_face_naive = NB_model_face.predict(XFaceValidation)
predicted_test_Face_naive = NB_model_face.predict(XFaceTest)
visualize(faceTest, predicted_test_Face_naive,"samples of face test dataset with its prediction for naive bayes")
print("the accuracy of the digits dataset for the naive bayes classifier is ")
print(f"the training accuracy is {accuracy_score(y_face_train, predicted_train_face_naive) * 100} %")
print(f"the validation accuracy is {accuracy_score(y_face_validation, predicted_validation_face_naive) * 100} %")
print(f"the test accuracy is {accuracy_score(y_face_test, predicted_test_Face_naive) * 100} %")
print("------------------------------------------------------------------------")

# ------------------------------------------------------------------------------------#

# building the KNN claseifier
from sklearn.neighbors import KNeighborsClassifier

best_digit_k = {}
for i in range(2, 10):
    KNN_digits = KNeighborsClassifier(n_neighbors=i)
    KNN_digits.fit(XDigitsTrain, y_digits_train)
    KNN_digits_train_accuracy = accuracy_score(y_digits_train, KNN_digits.predict(XDigitsTrain))
    KNN_digits_validation_accuracy = accuracy_score(y_digits_validation, KNN_digits.predict(XDigitsValidation))
    KNN_digits_test_accuracy = accuracy_score(y_digits_test, KNN_digits.predict(XDigitsTest))
    best_digit_k[i] = KNN_digits_validation_accuracy

bestDK = max(best_digit_k, key=best_digit_k.get)
accuracy_change(best_digit_k,"K vs validation accuracy for digit dataset")
print(f"best k for digit data set is {bestDK}")
KNN_digits = KNeighborsClassifier(n_neighbors=bestDK)
KNN_digits.fit(XDigitsTrain, y_digits_train)
KNN_digits_train_accuracy = accuracy_score(y_digits_train, KNN_digits.predict(XDigitsTrain))
KNN_digits_validation_accuracy = accuracy_score(y_digits_validation, KNN_digits.predict(XDigitsValidation))
KNN_digits_test_accuracy = accuracy_score(y_digits_test, KNN_digits.predict(XDigitsTest))
visualize(x_digits_test, KNN_digits.predict(XDigitsTest),"samples of digits test dataset with its prediction for KNN")
print(f"the accuracy of the digits dataset for k = {bestDK} for the KNN classifier is ")
print(f"the training accuracy is {KNN_digits_train_accuracy * 100} %")
print(f"the validation accuracy is {KNN_digits_validation_accuracy * 100} %")
print(f"the test accuracy is {KNN_digits_test_accuracy * 100} %")
print("------------------------------------------------------------------------")

# KNN for faces dataset
best_face_k = {}
for i in range(2, 10):
    KNN_face = KNeighborsClassifier(n_neighbors=i)
    KNN_face.fit(XFaceTrain, y_face_train)
    KNN_face_train_accuracy = accuracy_score(y_face_train, KNN_face.predict(XFaceTrain))
    KNN_face_validation_accuracy = accuracy_score(y_face_validation, KNN_face.predict(XFaceValidation))
    KNN_face_test_accuracy = accuracy_score(y_face_test, KNN_face.predict(XFaceTest))
    best_face_k[i] = KNN_face_validation_accuracy

bestFK = max(best_face_k, key=best_face_k.get)
accuracy_change(best_face_k,"K vs validation accuracy for face dataset")
print(f"best k for face data set is {bestFK}")
KNN_face = KNeighborsClassifier(n_neighbors=bestFK)
KNN_face.fit(XFaceTrain, y_face_train)
KNN_face_train_accuracy = accuracy_score(y_face_train, KNN_face.predict(XFaceTrain))
KNN_face_validation_accuracy = accuracy_score(y_face_validation, KNN_face.predict(XFaceValidation))
KNN_face_test_accuracy = accuracy_score(y_face_test, KNN_face.predict(XFaceTest))
visualize(faceTest, KNN_face.predict(XFaceTest),"samples of face test dataset with its prediction for KNN")
print(f"the accuracy of the face dataset for k = {bestFK} for the KNN classifier is ")
print(f"the training accuracy is {KNN_face_train_accuracy * 100} %")
print(f"the validation accuracy is {KNN_face_validation_accuracy * 100} %")
print(f"the test accuracy is {KNN_face_test_accuracy * 100} %")
print("------------------------------------------------------------------------")
