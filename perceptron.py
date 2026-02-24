#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 1: Voted Perceptron
"""

import numpy as np
import pandas as pd
import sys

def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
   #shape = (n samples, 57 features)

    df_X = pd.read_csv(X_path)
    X = df_X.values                  # numpy array

    if y_path is not None:           #ensure there is a label; should be ok for given data; perhaps hidden test cases are diff?
        df_y = pd.read_csv(y_path)
        y = df_y.values.flatten()       #read in as a 2D table, (n, 1) => (n, )
        return X, y

    return X

    #raise NotImplementedError


def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    #X shape: (3680, 57); y shape: (3680,) First 5 labels: [-1 -1 -1  1 -1]
    #adding bias --> X shape is now (3680, 58) --> x' = [x1, x2, ... x57, 1]

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return X_train, X_test

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


class VotedPerceptron:
    """Voted Perceptron Classifier."""

    def train(self, X, y, epochs = 10):
        """Fit the classifier to training data."""
        self.weights = []       #init to empty
        self.counts = []
        #for each training point if yi(w * xi) > 0 --> increment count bc this is correct
        #else, store current w, c and update weight --> w = w + yi * xi ;    resent count to 1
        #end, store final x, y

        nSamples, nFeatures = X.shape
        c = 0
        w = np.zeros(nFeatures)

        for e in range(epochs):
            for i in range(nSamples):
                if y[i] * np.dot(w, X[i]) > 0:
                    c += 1
                else:
                    if c > 0:
                        self.weights.append(w.copy())
                        self.counts.append(c)
                    w = w + y[i] * X[i]
                    c = 1


        #final weight --> store
        if c > 0:
            self.weights.append(w.copy())   #need copy so it doesn't reference same obj
            self.counts.append(c)


        
    def predict(self, X):
        """Predict labels for input samples."""
        #sign(sum(ck * sign(wk * x)) )
        #wk = stored weight vector, ck = how long the weight survived, final pred = weighted vote
        # compute dot product w every stored , take sign, mult by count, sum all votes, take final sign 
        predictions = []

        for x in X: 
            sum = 0
            for w, c in zip(self.weights, self.counts):
                sum += c * np.sign(np.dot(w, x))        #perceptron dec
            
            if(sum >= 0):
                predictions.append(1)           #count 0 as positive 
            else:
                predictions.append(-1)  

        return np.array(predictions)
        #raise NotImplementedError


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)        #use avg

    #raise NotImplementedError


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder.""" #auto grader will provide test and train data; no need to split
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test = load_data(test_data_file)  

    X_train, X_test = preprocess_data(X_train, X_test)

    currVotedPercep = VotedPerceptron()
    currVotedPercep.train(X_train, y_train)

    y_pred = currVotedPercep.predict(X_test)

    pd.DataFrame(y_pred).to_csv(pred_file, index=False, header=False)       #same format as spam_y; no col/row num 
    
    #raise NotImplementedError

def runForReport(X, y):
    #for report --> use last 10% for training data
    #from the 90% of training data, pick 1, 2, 5, 10, 20, and 100% to train and compare results
    #plot accuracy as a function of size of fraction 
   
   # X_train, y_train = load_data(Xtrain_file, Ytrain_file)      #splitting the training data into 90-10 split; using the 10% as test
    X_train, y_train, X_test, y_test = splitData(X, y)

    #adding bias (after split)
    X_train, X_test = preprocess_data(X_train, X_test)
    n_train = X_train.shape[0]

    fractions = [.01, .02, .05, .1, .2, 1.0]
    accuracies = []

    #training & eval on subset
    for f in fractions:
        n_red = int(n_train * f)
        X_train_red = X_train[:n_red]
        y_train_red = y_train[:n_red]

        currVotedPercep = VotedPerceptron()
        currVotedPercep.train(X_train_red, y_train_red)

        y_pred = currVotedPercep.predict(X_test)        #predicting test data now

        currAccuracy = evaluate(y_test, y_pred)
        accuracies.append(currAccuracy)
        print(f"{f} : {currAccuracy}")

    return accuracies



# if __name__ == "__main__":
    #  testing
#    if len(sys.argv) != 5:
#     print("Wrong num args")
#     sys.exit(1)

    # X, y = load_data("spam_X.csv", "spam_y.csv")
    # runForReport(X, y)

