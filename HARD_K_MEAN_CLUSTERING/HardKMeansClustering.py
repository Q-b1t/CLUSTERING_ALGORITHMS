import numpy as np
import pandas as pd
from tqdm import tqdm

class HardKMeansClustering:
  def __init__(self,max_iterations):
    self.max_iterations = max_iterations


  def fit(self,X,Y):
    N,D = X.shape # retrieve the number of samples and the number of features
    K = len(set(Y)) # retrieve the number of clusters

    # centroid initialization
    cluster_centers = np.zeros((K,D))
    # randomly select the centroids
    for k in range(K):
      i = np.random.choice(N)
      cluster_centers[k] = X[i]


    self.cluster_identities = np.zeros(N)
    self.saved_cluster_identities = list() # placeholder for saving the identities state (for checking for convergence)
    min_dists = np.zeros(N)
    self.costs = list()

    for i in tqdm(range(self.max_iterations)):
      # save the state
      old_cluster_identities = self.cluster_identities.copy()
      self.saved_cluster_identities.append(old_cluster_identities)


      # detemine the cluster identities
      for n in range(N):
        # initialize the search parameters
        clossest_k = -1
        min_dist = float("inf")
        for k in range(K):
          # compute the euclidean distance between the sample in question and the mean
          dist = (X[n]-cluster_centers[k]).dot(X[n]-cluster_centers[k])
          if dist < min_dist:
            min_dist = dist
            clossest_k = k
        self.cluster_identities[n] = clossest_k
        min_dists[n] = min_dist

      self.costs.append(min_dists.sum())

      # recalculate the means
      for k in range(K):
        cluster_centers[k,:] = X[self.cluster_identities == k].mean(axis = 0)
        #print(X[self.cluster_identities == k].mean(axis = 0))

      # check for convergeance
      if np.all(old_cluster_identities == self.cluster_identities):
        print(f"[+] Converged on step: {i}")
        break

  def get_cluster_identities(self):
    """
    Retrieve and array of labels sequentially corresponding to the features provided in
    the training phase (returns Y according to the data clustering).
    """
    return self.cluster_identities

  def get_saved_cluster_identities(self):
    """
    Retrieve the saved states of the cluster identities to see the clustering progress.
    """
    return self.saved_cluster_identities

  def get_costs(self):
    """
    Get the historical data of costs computer across all the training iterations
    """
    return self.costs