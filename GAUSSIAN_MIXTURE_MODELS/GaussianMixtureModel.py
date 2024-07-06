import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class GaussianMixtureModel:
  def fit(self,X,K,max_iterations = 20,smoothing = 1e-02):
    """
    This is a gaussian mixture model implementation class.
    Arguments:
      - X: Clustering Dataset. It is a matrix of N x D where N stands for the number of samples and D for the number of features
      - K: The number of desired clusters
      - max_iterations: The number of times the expectation maximization algorithm will be run
      - smoothing: Smoothing term for gaining numberical stability during the covariance computation (to avoid dividing by zero)
    """
    # initialize parameters
    self.N,self.D = X.shape # retrieve the number of samples and features respectively
    self.K = K # number of clusters
    # initialize important matrices for training
    self.M = np.zeros((self.K,self.D)) # means across all clustering and all the features
    self.R = np.zeros((self.N,self.K)) # responsability (weighted mean calculation based on how likely is sample n to belong to cluster k)
    self.C = np.zeros((self.K,self.D,self.D)) # initialize the covariance tensor, which consists of grouping K covariance matrices together. The covariance is calculated with the features
    self.PI = np.ones(self.K) / self.K # initialize PI (not matematical quantity but rather weight vector) to a uniform distribution

    # initialize simulation parameters
    self.max_iterations = max_iterations
    self.smoothing = smoothing

    # expectation maximization
    self.log_likelihood_estimations = list() # placeholder to store log likelyihood estimates
    self.weighted_pdfs = np.zeros((self.N,self.K)) # placeholder to store the probability density funtions per sample and cluster

    # initialization
    for k in range(self.K):
      self.M[k] = X[np.random.choice(self.N)] # initialize the means by selecting a random sample value
      self.C[k] = np.eye(self.D) # intialize the covariance tensor's layers to the identity matrix

    for i in tqdm(range(self.max_iterations)):
      # E-step: calculate the responsabilities
      for k in range(self.K):
        self.weighted_pdfs[:,k] = self.PI[k] * multivariate_normal.pdf(X,self.M[k],self.C[k]) # get the probability density functions by sampling from a multivariate normal distribution modeled after cluster k params
      self.R = self.weighted_pdfs / self.weighted_pdfs.sum(axis = 1, keepdims = True) # keep the matrix bidimentional

      # M-step: Recalculate the means
      for k in range(self.K):
        N_k = self.R[:,k].sum() # get the sum across all the gammas
        self.PI[k] = N_k / self.N # calculate the weights of how much each sample contributes to the cluster
        self.M[k] = self.R[:,k].dot(X) / N_k # recalculate the means by multiplying the responsabilities times the samples


        # calculate the "gradient" (diffrence of X and the means for cluster k)
        delta = X - self.M[k]
        R_delta = np.expand_dims(self.R[:,k],-1) * delta
        # we recalculate the covariance using the gradient
        self.C[k] = R_delta.T.dot(delta) / N_k + np.eye(self.D)*self.smoothing

      # compute the log likelyhood
      log_likelihood = np.log(self.weighted_pdfs.sum(axis = 1)).sum()
      self.log_likelihood_estimations.append(log_likelihood)

      if i > 0:
        if np.abs(self.log_likelihood_estimations[i] - self.log_likelihood_estimations[i-1]) < 0.1:
          break

    return self.R,self.log_likelihood_estimations
