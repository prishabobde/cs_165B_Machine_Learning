#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 2: Support Vector Machine (SVM)
"""

import numpy as np
import pandas as pd


def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    df_X = pd.read_csv(X_path)      #same as percep
    X = df_X.values                  # numpy array

    if y_path is not None:           #ensure there is a label; should be ok for given data; perhaps hidden test cases are diff?
        df_y = pd.read_csv(y_path)
        y = df_y.values.flatten()       #read in as a 2D table, (n, 1) => (n, )
        return X, y

    return X
    #raise NotImplementedError


def splitData(X, y):
    #shuffle; prev accuracy ~ 40%; maybe split wise it was clustering classes; shuff improves accuracy
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    n = X.shape[0]
    splitIdx = int(0.9 * n)       # 90% and 10%
    X_train = X[:splitIdx]
    y_train = y[:splitIdx]

    X_test = X[splitIdx:]
    y_test = y[splitIdx:]

    return X_train, y_train, X_test, y_test




def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    #X shape: (3680, 57); y shape: (3680,) First 5 labels: [-1 -1 -1  1 -1]
    #adding bias --> X shape is now (3680, 58) --> x' = [x1, x2, ... x57, 1]

    #mean = 0; var = 1  --> normalize
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1        # prev division by zero
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
   
   #bias
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return X_train, X_test


    #raise NotImplementedError


class SVMClassifier:
    """Support Vector Machine Classifier."""


    def train(self, X, y, lambda_val=0.001, learning_rate=0.01, epochs=200):
        """Fit the classifier to training data."""
        nSamples, nFeatures = X.shape
        w = np.zeros(nFeatures)
        initial_learning_rate = learning_rate

        #learning rate = large in the beginning, towards the end decrease

        for e in range(epochs):

            indices = np.random.permutation(nSamples)       #trying to shuffle more
            for i in indices:
                if (y[i] * np.dot(w, X[i])) < 1:
                    w = w + learning_rate* (y[i] * X[i] - lambda_val * w)
                else:
                    w = w - learning_rate * lambda_val * w
        
        self.w = w
        #raise NotImplementedError

    def predict(self, X):
        """Predict labels for input samples."""
        scores = np.dot(X, self.w)
        predictions = np.sign(scores)
        predictions[predictions == 0] = 1       #avoid 0s
        return predictions

        #raise NotImplementedError


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)        #use avg
    #raise NotImplementedError


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    #same as percept
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test = load_data(test_data_file)  

    X_train, X_test = preprocess_data(X_train, X_test)

    currSVCM = SVMClassifier()
    currSVCM.train(X_train, y_train, lambda_val=0.0001, learning_rate=0.01, epochs=200)

    y_pred = currSVCM.predict(X_test)

    pd.DataFrame(y_pred).to_csv(pred_file, index=False, header=False)       #same format as spam_y; no col/row num 
   # raise NotImplementedError


def runForReport(X, y):
    #for report --> use last 10% for training data
    #plot accuracy as a function of size of lambda --> use log scale 
   
   # X_train, y_train = load_data(Xtrain_file, Ytrain_file)      #splitting the training data into 90-10 split; using the 10% as test
    X_train, y_train, X_test, y_test = splitData(X, y)

    #adding bias (after split)
    X_train, X_test = preprocess_data(X_train, X_test)
    n_train = X_train.shape[0]

    lam_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    accuracies = []

    #training & eval on subset
    for l in lam_vals:

        currSVCM = SVMClassifier()
        currSVCM.train(X_train, y_train, lambda_val=l, learning_rate=0.01, epochs=100)

        y_pred = currSVCM.predict(X_test)        #predicting test data now

        currAccuracy = evaluate(y_test, y_pred)
        accuracies.append(currAccuracy)
        print(f"{l} : {currAccuracy}")

    return accuracies


if __name__ == "__main__":
    X, y = load_data("spam_X.csv", "spam_y.csv")
    runForReport(X, y)