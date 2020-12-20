import os
import pickle
import gzip
import argparse
import numpy as np

from knn import KNN
import linear_model
from neural_net1 import NeuralNet

from sklearn.preprocessing import LabelBinarizer

def load_dataset(filename):
    with gzip.open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f, encoding="latin1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        train_set, valid_set, test_set = load_dataset('mnist.pkl.gz')
        X, y = train_set
        Xtest, ytest = test_set
        X_val, y_val = valid_set

        min_valError = np.inf
        min_k = np.inf
        for i in range(11):
            # Initialize model
            model = KNN(i)
            model.fit(X, y)

            yhat = model.predict(X_val)
            valError = np.mean(yhat != y_val)
            print("Val error     = ", valError)

            if valError < min_valError:
                min_valError = valError
                min_k = i

        print("Min valError     = ", min_valError)
        print("Min k     = ", min_k)

        model = KNN(min_k)
        model.fit(X, y)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)


    elif question == '2':
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set
        Xval, yval = valid_set

        min_error = np.inf
        min_max_iter = np.inf

        # hyperparameter tuning: hidden size, lammy, max eval
        for i in range(3, 11):
            max_eval = 100 * i

            model = linear_model.softmaxClassifier(maxEvals=max_eval)
            model.fit(X, y)

            yhat = model.predict(Xval)
            valError = np.mean(yhat != yval)
            print("Val error and params     = ", [valError, max_eval])

            if valError < min_error:
                min_error = valError
                min_max_iter = max_eval

        print("Minimized combination ", [min_error, min_max_iter])

        # Initialize model
        model = linear_model.softmaxClassifier(maxEvals=min_max_iter)
        model.fit(X, y)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    elif question == '3':
        train_set, valid_set, test_set = load_dataset('mnist.pkl.gz')
        X, y = train_set
        Xtest, ytest = test_set
        Xval, yval = valid_set

        # min_error = np.inf
        # min_lambda = np.inf
        # min_max_iter = np.inf
        #
        # # hyperparameter tuning: hidden size, lammy, max eval
        # for i in range(3, 11):
        #     for j in range(1, 6):
        #         max_eval = 100 * i
        #         lammy = 0.005 * j
        #
        #         model = linear_model.SVM(maxEvals=max_eval, lammy=lammy)
        #         model.fit(X, y)
        #
        #         yhat = model.predict(Xval)
        #         valError = np.mean(yhat != yval)
        #         print("Val error and params     = ", [valError, max_eval])
        #
        #         if valError < min_error:
        #             min_error = valError
        #             min_lambda = lammy
        #             min_max_iter = max_eval
        #
        # print("Minimized combination ", [min_error, min_lambda, min_max_iter])

        # Initialize model
        model = linear_model.SVM(maxEvals=500, lammy=0.01)
        model.fit(X, y)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    elif question == '4':
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set
        Xval, yval = valid_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        # min_error = np.inf
        # min_hidden_size = np.inf
        # min_lammy = np.inf
        # min_max_iter = np.inf

        # # hyperparameter tuning: hidden size, lammy, max eval
        # for i in range(4, 11):
        #     for j in range(1, 5):
        #         for k in range(1, 6):
        #             max_eval = 100 * i
        #             hidden_size = [50 * j]
        #             lammy = 0.005 * k

        #             model = NeuralNet(hidden_layer_sizes=hidden_size, lammy=lammy, max_iter=max_eval)
        #             model.fit(X, Y)

        #             yhat = model.predict(Xval)
        #             valError = np.mean(yhat != yval)
        #             print("Val error and params     = ", [valError, max_eval, hidden_size, lammy])

        #             if valError < min_error:
        #                 min_error = valError
        #                 min_hidden_size = hidden_size
        #                 min_lammy = lammy
        #                 min_max_iter = max_eval

        # print("Minimized combination ", [min_error, min_hidden_size, min_lammy, min_max_iter])

        # Initialize model
        model = NeuralNet(hidden_layer_sizes=[100], lammy=0.01, max_iter=1000)
        model.fit(X, Y)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    else:
        print("Unknown question: %s" % question)
