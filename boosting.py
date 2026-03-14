#!/usr/bin/env python3
"""
CMPSC 165B - Machine Learning
Homework 3, Problem 2: Boosting Classifier
"""

import numpy as np
import pandas as pd


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
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1       #prev div by zero
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    #bias 
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return X_train, X_test


class BoostingClassifier:
    """AdaBoost Classifier with weighted linear classifier as weak learner."""

    def weak_predict(self, X, midpoint, direction):     #h(x)
        scores = (X - midpoint) @ direction
        pred = np.sign(scores)
        pred[pred == 0] = 1

        return pred

    def train(self, X, y, T=10):
        """Fit the classifier to training data."""
        self.models = []           #weak learners
        self.alphas = []        #weights alpha t
        self.T = T         #change T as needed

        #init w = 1/m for all i = 1,...,m (m=num training points)
        m = X.shape[0]

        w = np.ones(m)/m

         #need a mask; otherwise it will pick 1rst elem
        positives = (y==1)
        negatives = (y == -1)


        for t in range(self.T):        #each t --> 1 weak learner
           

            exemplar_plusOne = np.sum(w[positives][:,None] * X[positives], axis=0) / np.sum(w[positives])       #weighted mean sum wi*Xi / wi
            exemplar_minusOne = np.sum(w[negatives][:,None] * X[negatives], axis=0) / np.sum(w[negatives])

            midpoint = (exemplar_plusOne + exemplar_minusOne) / 2
            direction = exemplar_plusOne - exemplar_minusOne

            pred = self.weak_predict(X, midpoint, direction)

            #weighted error
            misclassified = (pred != y)
            epsilon_t = np.sum(w[misclassified])           #add all misclassified
            epsilon_t = max(epsilon_t, 1e-10)           #in case it's 0 prev crash

            if epsilon_t >= 0.5:        #if learner is >= .5 it's worse than random --> hurts boosting
                break

            #compute model weight
            alpha_t = 0.5 * np.log((1 - epsilon_t ) / epsilon_t )


            #udpate sample weights
            for i in range(m):
                if pred[i] != y[i]:              # misclassified
                    w[i] *= np.exp(alpha_t)
                else:                            # correct
                    w[i] *= np.exp(-alpha_t)

        
            #normalize
            w = w / np.sum(w)

            #storing
            self.models.append((midpoint, direction))
            self.alphas.append(alpha_t)


    def predict(self, X):
        """Predict labels for input samples."""
        
        total = np.zeros(X.shape[0])

        for alpha_t, model in zip(self.alphas, self.models):
            midpoint, direction = model

            pred = self.weak_predict(X, midpoint, direction)

            total += alpha_t * pred

        final = np.sign(total)
        final[final == 0] = 1           #make 0 --> +1
        
        return final




def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)
    


def runForReport(X, y):
    X_train, y_train, X_test, y_test = splitData(X, y)

    #adding bias (after split)
    X_train, X_test = preprocess_data(X_train, X_test)
    n_train = X_train.shape[0]

    boosting_vals = [1, 3, 5, 10, 20, 50]

    for b in boosting_vals:
        model = BoostingClassifier()
        model.train(X_train, y_train, T=b)

        predictions = model.predict(X_test)
        accuracy = evaluate(y_test, predictions)

        print("Boosting val: ", b, " ; accuracy: ", accuracy)



def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""


    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test = load_data(test_data_file)  

    X_train, X_test = preprocess_data(X_train, X_test)

    model = BoostingClassifier()
    model.train(X_train, y_train)
    
    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred).to_csv(pred_file, index=False, header=False)



# if __name__ == "__main__":
#     X, y = load_data("wine_X.csv", "wine_y.csv")
#     runForReport(X, y)

