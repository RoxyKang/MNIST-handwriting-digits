import numpy as np
from numpy.linalg import solve, norm
import findMin
from scipy.optimize import approx_fprime
import utils
from neural_net import log_sum_exp

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.L0_lambda * np.count_nonzero(w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the selected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature
                w, f = minimize(list(selected_new))
                if f < minLoss:
                    minLoss = f
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class logRegL1(logReg):
    def __init__(self, L1_lambda=1.0, maxEvals=400, verbose=1):
        self.lammy = L1_lambda
        self.maxEvals = maxEvals
        self.verbose = verbose

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.lammy,
                                      self.maxEvals, X, y, verbose=self.verbose)

class logRegL2(logReg):
    # L2 Regularized Logistic Regression
    def __init__(self, lammy=1.0, maxEvals=400, verbose=1):
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.verbose = verbose

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy/2 * (norm(w,2) **2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class logLinearClassifier(logReg):
    def fit(self,X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1
            # utils.check_gradient(self, X, ytmp)
            # print(X.shape, w[i].shape, ytmp.shape)

            (self.w[i], f) = findMin.findMin(self.funObj, self.w[i],
                                        self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X@self.w.T, axis=1)

class softmaxClassifier:
    # Q3.4 - Softmax for multi-class classification
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes
        W = np.reshape(w, (k,d))
        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1
        XW = np.dot(X, W.T)
        # Z = np.sum(np.exp(XW), axis=1)
        max_per_row = np.max(XW,axis=1)
        tmp = np.sum(np.exp(XW-max_per_row[:,None]), axis=1)
        # Calculate the function value
        f = - np.sum(XW[y_binary] - log_sum_exp(XW))
        # Calculate the gradient value
        # g = np.exp(XW-max_per_row[:,None]) / tmp[:,None] - y_binary
        g = (np.exp(XW-max_per_row[:,None]) / tmp[:,None] - y_binary).T@X
        return f, g.flatten()

    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k
        self.W = np.zeros(d*k)
        self.w = self.W # because the gradient checker is implemented in a silly way
        # Initial guess
        # utils.check_gradient(self, X, y)
        (self.W, f) = findMin.findMin(self.funObj, self.W, self.maxEvals, X, y, verbose=self.verbose)
        self.W = np.reshape(self.W, (k,d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class SVM(softmaxClassifier):
    def __init__(self, maxEvals=400, verbose=0, lammy=0.01):
        self.maxEvals = maxEvals
        self.verbose = verbose
        self.lammy = lammy


    def funObj(self, w, X, y):
        n, d = X.shape
        # reshape w into kxd matrix
        w = np.reshape(w, (self.n_classes, d))

        f = 0
        g = np.zeros((self.n_classes, d))

        for i in range(n):
            for j in range(self.n_classes):
                if j == y[i]:
                    continue

                temp = 1-np.dot(w[y[i]].T, X[i])+np.dot(w[j].T, X[i])
                if temp > 0:
                    f += temp
                    g[y[i]] -= X[i]
                    g[j] += X[i]

        # Regularization
        f += self.lammy/2 * np.sum(norm(w, axis=1)**2)
        g += self.lammy * w

        g = g.flatten()

        return f, g
