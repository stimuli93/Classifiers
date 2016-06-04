"""
Created on Sat Jun  4 22:50:08 2016

@author: Saurabh
"""

import numpy as np
import scipy.spatial.distance as distance

class KNearestNeighbor(object):
  """ a kNN classifier  """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, dist='euclidean'):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - dist: Determines which distance to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if dist == 'euclidean':
      dists = self.compute_L2_distance(X)
    else:
      dists = distance.cdist(X,self.X_train,dist)  
    return self.predict_labels(dists, k=k)


  def compute_L2_distance(self, X):
    """
    Compute the euclidean distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Inputs
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
         
    Returns:
    - dists: A numpy array of shape (num_test,num_train) containing euclidean distance 
      between each pair of test/train points       
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    
    dists = np.zeros((num_test, num_train))
    prod = -2 * X.dot(self.X_train.T)
    sum1 = np.sum(X*X,axis=1)
    sum2 = np.sum(self.X_train*self.X_train,axis=1)
    
    prod = (prod.T + sum1).T
    dists = prod + sum2
    dists = np.sqrt(dists)
    return dists
    
      

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = self.y_train[np.argsort(dists[i])[:k]]  
      y_pred[i] = np.argmax(np.bincount(closest_y))

    return y_pred

