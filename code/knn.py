"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        # Compute cosine_distance distances between X and Xtest
        dist2 = utils.euclidean_dist_squared(X, Xtest)

        # sort the distances to other points
        inds = np.argsort(dist2, axis=0)[:self.k,:]

        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            # compute mode of k closest training pts
            yhat[i] = utils.mode(y[inds[:,i]])

        return yhat

        '''
         if there is any zero row in X1 or X2, set the distance between any zero row in X1 to all the rows X2 to zero or vice verca.
        '''

    def cosine_distance(self, X1, X2):
        N, D = X1.shape
        M, D = X2.shape

        dot_product = np.dot(X1, X2.T) # N X M

        X1_norm = np.linalg.norm(X1, axis=1)
        X2_norm = np.linalg.norm(X2, axis=1)

        norm_product = np.dot(X1_norm.reshape((N,1)), X2_norm.reshape((M,1)).T)
        dot_product /= norm_product

        return 1-dot_product
