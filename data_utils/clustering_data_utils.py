import numpy as np

def get_sample_cluster_data(N,D = 2):
  """
  Returns a sample dataset with 4 clusters. The idea behind this function is to 
  have a benchmark for testing unsupervised learning algorithms implementations
  for debugging and troubleshooting purposes.
  Parameters:
  - N: Number of samples required.
  - D: Number of features. It is recomended to use 2 as this is a dimentional 
  space that can be plotted. This makes easy to assess whether the implementation
  works.
  """
  # means around which the clusters will be created
  mu_1 = np.array([0,0])
  mu_2 = np.array([5,5])
  mu_3 = np.array([0,5])
  mu_4 = np.array([5,0])

  # create a placeholder for the training data
  X = np.zeros((N,D))

  # populate the dataset
  X[:100,:] = np.random.randn(100,D) + mu_1
  X[100:200,:] = np.random.randn(100,D) + mu_2
  X[200:300,:] = np.random.randn(100,D) + mu_3
  X[300:,:] = np.random.randn(100,D) + mu_4

  # labels
  Y = np.array([0]*100+[1]*100+[2]*100+[3]*100)

  return X,Y