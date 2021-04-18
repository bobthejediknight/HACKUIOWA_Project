"""
Title: WeightClassSVM.py
Author: Antonio Marino
Purpose: HackUIowa Submission

Summary:
This file trains a machine learning Support Vector Machine model
on weightlifting data in order to guess a given individual's weight.
First, the data is preprocessed by preprocess(filePath) and turned
into a numpy array. Then, the SVM model is initialized, fit to the
training data, and makes predictions on both the training and test
set. Finally, the results are printed to the terminal.

Below, are two sources that I referred to during the writing of
this program.

To run this file, navigate to the appropriate directory and
type python3 WeightClassSVM.py

1. https://docs.python.org/3/library/csv.html
This source is the python documentation on reading and writing
to a CSV file.

2. https://numpy.org/doc/stable/user/quickstart.html
This source describes how to initialize a numpy array

3. https://openpowerlifting.gitlab.io/opl-csv/bulk-csv-docs.html
This source contains the Open Powerlifting Dataset on which the model
was trained.
"""
import csv
from Preprocess import preprocess
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



if __name__ == "__main__":
    X, Y = preprocess('/Users/antoniomarino/Downloads/openipf-2021-03-09/openipf-2021-03-09-bbf05cdc.csv')

    features_train, labels_train, features_test, labels_test = np.array(X[0:1000]), np.array(Y[0:1000]), np.array(X[1000:]), np.array(Y[1000:])



    clf = SVC(kernel = "linear")

    clf.fit(features_train, labels_train)

    pred_train = clf.predict(features_train)

    accuracy_train = accuracy_score(pred_train, labels_train)

    pred_test = clf.predict(features_test)

    accuracy_test = accuracy_score(pred_test, labels_test)

    for row in labels_test:
        print("labels test:", labels_test[row], "predicted label:", pred_test[row])

    print("Accuracy of Machine Learning Model on training set:", accuracy_train)
    print("Accuracy of Machine Learning Model on test set:", accuracy_test)
