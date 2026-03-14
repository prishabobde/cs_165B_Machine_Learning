#!/usr/bin/env python3
"""
CMPSC 165B - Machine Learning
Homework 3, Problem 1: K-Nearest Neighbors
"""

import numpy as np
import pandas as pd
import math




def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    df_X = pd.read_csv(X_path)
    X = df_X.values                 #numpy arr

    if y_path is not None:
        df_y = pd.read_csv(y_path)
        y = df_y.values.flatten()
        return X,y

    return X

def splitData(X, y):
    #add shuffle
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

    #features are at different scales --> need to normalize

    #normalize? try without first and then implement next
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1       #prev div by zero
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    #bias 
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return X_train, X_test

    #TODO: Implement
    #raise NotImplementedError

def computeDistance(x, x_prime):
    diff = (x - x_prime)
    return math.sqrt(np.sum(diff ** 2))
    


class KNNClassifier:
    """K-Nearest Neighbors Classifier."""

    def train(self, X, y):
        """Fit the classifier to training data."""
        #compute distance from x to every training point
        #select k nearest training points
        #predict most common label among k neighbors    
            #tie breaking: 
                # if there is a tie in votes; choose label of closest neighbor among tied labels
                #if tie in distance --> choose lower label (ex choose -1 over 1)

        self.X_train = X
        self.y_train = y

        m = X.shape[0]      #num training points
        w = np.ones(m) / m      #init all to 1/m for i = 1,..., m
       

        # TODO: Implement
        #raise NotImplementedError

    def predict(self, X):
        """Predict labels for input samples."""
        #which is the most common label among k neighbors

        predictions = []

        for x in X:
            distances = []

            for i in range(len(self.X_train)):
                d = computeDistance(x, self.X_train[i])
                distances.append((d, self.y_train[i]))              #saving the label too


            #sorting
            sorted_distances = sorted(distances, key=lambda t: (t[0], t[1]))        #sorting by dist. label (so -1 picked over 1)

            neighbors = sorted_distances[:self.k]       #pick k neighbots

            #majority
            labels = [label for (_, label) in neighbors]

            values, counts = np.unique(labels, return_counts=True)

            maxVal = np.max(counts)

            #checking if items are tied
            tied = values[counts == maxVal]

            if len(tied) == 1:
                majority = tied[0]
            
            else:           #pick label of closest neighbor
                for (_, label) in neighbors:
                    if label in tied:
                        majority = label
                        break                       #sorted in order; pick the first one that works



            predictions.append(majority)

        return np.array(predictions)
        

        # TODO: Implement
        #raise NotImplementedError


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)

    # TODO: Implement
    #raise NotImplementedError


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""

    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test = load_data(test_data_file)  

    X_train, X_test = preprocess_data(X_train, X_test)

    model = KNNClassifier()
    model.k = 5
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    pd.DataFrame(y_pred).to_csv(pred_file, index=False, header=False)       #same format as spam_y; no col/row num 



def runForReport(X, y):
    X_train, y_train, X_test, y_test = splitData(X, y)

    #adding bias (after split)
    X_train, X_test = preprocess_data(X_train, X_test)
    n_train = X_train.shape[0]

    k_vals = [1, 3, 5, 7, 9, 11, 15, 21]


    for k in k_vals:
        model = KNNClassifier()
        model.k = k
        model.train(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = evaluate(y_test, predictions) 

        print("k = ", k, ", accuracy = ", accuracy)


   


if __name__ == "__main__":
    X, y = load_data("wine_X.csv", "wine_y.csv")
    runForReport(X, y)
  

