import numpy as np

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


  def fit(self,X, K, beta=3.0):
    N, D = X.shape
    exponents = np.empty((N, K))

    # initialize M to random
    initial_centers = np.random.choice(N, K, replace=False)
    M = X[initial_centers]

    costs = []
    k = 0
    for i in range(self.max_iterations):

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
    return M, R,costs