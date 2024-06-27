
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm

class SoftKMeansClustering:
  def __init__(self,max_iterations):
    """
    Arguments:
      - max_iterations: The number of max iterations to peform the cluster centroid search
    """
    self.max_iterations = max_iterations
  
  def distance(self,u,v):
    """
    This is an auxiliary function. It computes the distance between two vectors.
    """
    diff = u-v
    return diff.dot(diff)

  def cost(self,X,R,M):
    """
    It fits using a similar method than that of Hard K-means clustering, but this weights each sample's belonging
    to each cluster akin to a probability density function (all the weights are normalized to 1).
    Inputs:
     - X: Training samples
     - R: Responsability matrix/weights
     - M: Means
    """
    cost = 0
    for k in range(len(M)):
      diff = X - M[k]
      sq_distances = (diff * diff).sum(axis=1)
      cost += (R[:,k] * sq_distances).sum()
    return cost

  def purity(self,Y):
    """
    Implements the purity cost function.
    Parameters:
      - Y: Targets
      - R: Responsability Matrix (degree of confidence with which we can assume can assume a sample belongs to a cluster)
    """
    N, K = self.R.shape # retrieve the number of samples and clusters from the resposnabiity matrix's shape
    purity = 0.0 # initialize the purity
    for k in range(K):
      best_target = -1
      max_intersection = 0
      for j in range(K):
        intersection = self.R[Y==j, k].sum() # total sum of weights for which the target corresponds to the cluster
        if intersection > max_intersection:
          max_intersection = intersection
          best_target = j
      purity += max_intersection
    return purity / N 

  def purity_hard_labels(self,Y):
    """
    The implementation is the same as for the normal purity function with the slight 
    difference of calculating the intersection using only the samples of the most likely cluster instead 
    of every single cluster in the iteration.
    """
    # get explicit labels assigned from the cluster with the max weight
    C = np.argmax(self.R,axis = 1)
    N, K = self.R.shape # get the number of samples and labels (clusters)
    
    purity = 0.0
    for k in range(K): # iterate throough the clusters
      max_intersection = 0 # initialie the max intersection as zero since we have not yet look at any cluster
      for j in range(K): # iterate through the labels
        intersection = ((C == k) & (Y == j)).sum() # sum of the weights that belong to the same cluster and the same label 
        if intersection > max_intersection:
          max_intersection = intersection
      purity += max_intersection
    return purity / N
          

  def davies_boulding_index(self,X):
    """
    Calculates the Davies Boulding Index.
    Arguments:
      - X: Traning Data
      - R: Responsability Matrix (Probability of each sample belonging to each cluster)
      - M: Means obtained during the soft k means clustering training
    """
    N,D = X.shape # get the number of samples and features
    K,_ = self.M.shape # get the number of clusters

    # get the sigmas (standard deviations) note that since we do not know which sample belongs to which 
    # cluster, we calculate the standard deviations taking into account all the samples
    sigma = np.zeros(K)
    for k in range(K):
      diffs = X - self.M[k] # N x D
      squared_distances = (diffs * diffs).sum(axis = 1) # N
      weighted_square_distances = self.R[:,k] * squared_distances # literally take distances and multiply per the weight X * X
      sigma[k] = np.sqrt(weighted_square_distances.sum() / self.R[:,k].sum())
    
    # calculate the davies boudin index
    dbi = 0
    for k in range(K):
      max_ratio = 0
      for j in range(k):
        if k != j:
          numerator = sigma[k] + sigma[j]
          denominator = np.linalg.norm(self.M[k] - self.M[j])
          ratio = numerator / denominator
          if ratio > max_ratio:
            max_ratio = ratio
      dbi += max_ratio
    return dbi / K

  # davies boulding index
  def davies_boulding_index_hard(self,X):
    N,D = X.shape # get the number of samples and features
    _,K = self.R.shape # get the number of clusters

    # get the standard deviations and the means
    sigma = np.zeros(K)
    M = np.zeros((K,D))
    assignments = np.argmax(self.R, axis=1)
    for k in range(K):
      X_k = X[assignments == k]
      M[k] = X_k.mean(axis = 0)
      n = len(X_k)
      diffs = X_k-M[k]
      squared_differences = diffs * diffs
      sigma[k] = np.sqrt( squared_differences.sum() / n )
    # calculate the davies boulding index
    dbi = 0
    for k in range(K):
      max_ratio = 0
      for j in range(k):
        if k != j:
          numerator = sigma[k] + sigma[j]
          denominator = np.linalg.norm(M[k] - M[j])
          ratio = numerator / denominator
          if ratio > max_ratio:
            max_ratio = ratio
      dbi += max_ratio
    return dbi / K 

  def fit(self,X, K, beta=3.0):
    N, D = X.shape
    exponents = np.empty((N, K))

    # initialize M to random
    initial_centers = np.random.choice(N, K, replace=False)
    M = X[initial_centers]

    costs = []
    k = 0
    for i in tqdm(range(self.max_iterations)):

        k += 1
        # step 1: determine assignments / resposibilities
        for k in range(K):
            for n in range(N):
                exponents[n,k] = np.exp(-beta*self.distance(M[k], X[n]))
        R = exponents / exponents.sum(axis=1, keepdims=True)


        # step 2: recalculate means
        M = R.T.dot(X) / R.sum(axis=0, keepdims=True).T

        c = self.cost(X, R, M)
        costs.append(c)
        if i > 0:
            if np.abs(costs[-1] - costs[-2]) < 1e-5:
                break

        if len(costs) > 1:
            if costs[-1] > costs[-2]:
                pass

    #print(f"[~] Final Cost {costs[-1]}")
    self.M,self.R,self.costs = M,R,costs
    return M, R,costs
  
  def plot_clustering_results(self,X,K = None):
    """
    Plots the training points (they need to be a R2 vector space) with a color gradient,
    which varies on which cluster does it belongs to according to the model
    Arguments:
      - X: training data
      - K: number of cluster. It will be infered from the responsability matrix's dimentions
        if it is not provided
    """
    if K is None:
      _,K = self.R.shape
    random_colors = np.random.random((K,3))
    colors = self.R.dot(random_colors)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(self.costs)
    plt.title("Costs")
    plt.subplot(1,2,2)
    plt.scatter(X[:,0], X[:,1], c=colors)