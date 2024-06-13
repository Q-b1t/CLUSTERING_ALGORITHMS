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



# test for failure

def different_density_clusters():
  """
  Return a dataset of 1000 samples where a small subset is sampled from a different density.
  """
  X = np.zeros((1000, 2))
  X[:950,:] = np.array([0,0]) + np.random.randn(950, 2)
  X[950:,:] = np.array([3,0]) + np.random.randn(50, 2)
  return X


def different_density_clusters():
  """
  Return a dataset of 1000 samples where a small subset is sampled from a different density.
  """
  X = np.zeros((1000, 2))
  X[:950,:] = np.array([0,0]) + np.random.randn(950, 2)
  X[950:,:] = np.array([3,0]) + np.random.randn(50, 2)
  return X


def donut_clusters(n = 1000,cluster_number = 4, radius_distance = 10):
  """
  Create n clusters in the form of donut with a specified separation between the circunference of each cluster.
  Literally create instances of the problem that logistic regression, perceptor and every linear classifier fails to classify.
  Arguments: 
    - n: number to total samples (from all the clusters)
    - cluster_number: number of cluster. The idea is to create a balanced dataset. Therefore, each cluster has the same number of samples
    - radius_distance: the distance between each cluster starting from the center (distance between the donuts´ circunfenrences)
  """
  D = 2 # number of features is hardcoded because this is a dimention that we can plot

  samples_per_cluster = n // cluster_number # get the number of samples assigned to each cluster
  clusters = list() # placeholder to store all the clusters 
  radius = radius_distance # initialize for the first cluster to be at radius_distance from the center
  for c in range(cluster_number):
    r_c = np.random.randn(samples_per_cluster) + radius # sample data at radius distance from the center
    theta = 2*np.pi*np.random.random(samples_per_cluster) # sample different angles in the range of [0:2PI]
    x_cluster = np.concatenate( # transform the data by creating the coordinates
        [
            [r_c * np.cos(theta)],
            [r_c * np.sin(theta)]
        ]
    ).T
    clusters.append(x_cluster)
    radius += radius_distance # update the distance for generating the next cluster
  X = np.concatenate(clusters)
  return X
