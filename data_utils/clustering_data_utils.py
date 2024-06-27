import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def get_sample_cluster_data(D = 2):
  """
  Returns a sample dataset with 4 clusters. The idea behind this function is to 
  have a benchmark for testing unsupervised learning algorithms implementations
  for debugging and troubleshooting purposes.
  Parameters:
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
  X = np.zeros((400,D))

  # populate the dataset
  X[:100,:] = np.random.randn(100,D) + mu_1
  X[100:200,:] = np.random.randn(100,D) + mu_2
  X[200:300,:] = np.random.randn(100,D) + mu_3
  X[300:,:] = np.random.randn(100,D) + mu_4

  # labels
  Y = np.array([0]*100+[1]*100+[2]*100+[3]*100)

  return X,Y



# test for failure

def elongated_clusters(axis = 0):
  """
  Creates a dataset with elongated distributions (make half the data wider with respect to one axis)
  Arguments:
    - axis: the axis to elongate the samples with respect to
      - 0: elongate with respect to the vertical axis
      - 1: elongate with respect to the horizontal axis
  """
  changed_mean = [5, 0] if axis == 0 else [0,5]
  X = np.zeros((1000, 2))
  X[:500,:] = np.random.multivariate_normal([0, 0], [[1, 0], [0, 20]], 500) # sample from the mean 0
  X[500:,:] = np.random.multivariate_normal(changed_mean, [[1, 0], [0, 20]], 500) # keep the same covariance matrix, change the mean
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


# the same function present in my repository for unsupervised learning. I put it here for mere convenience when importing dependencies for my stuff.
def get_mnist(data_path,lower_limit = None,shuffle_data = False):
    """
    The function extracts the training data from the kaggle digit recognizer dataset.
    Arguments
        data_path: the full path of the file we wish to process. The code is thought to work in kaggle's dataset format
        lower_limit: float between 0 and 1 denoting the percentage of the data we wish to extract
        shuffle_data: boolean value denoting whether to shuffle or not the data. The convention is to shuffle for the training set
    """
    shuffle_data = float(shuffle_data)
    dataset = pd.read_csv(data_path)
    data = dataset.to_numpy()
    # dataset shape:  (42000, 785) where columns 0 are the labels and col [1:785] are pixel values
    # divide dataset into features and labels
    X,y = data[:,1:],data[:,0]
    X = X / 255.0 # normalize the data between 0 and 1
    # shuffle the data
    if shuffle_data:
        X,y = shuffle(X,y)
    if lower_limit is not None:
        # verify the data
        if lower_limit > 0.0 and lower_limit <= 1.0:
            threshold = len(data) * lower_limit
            threshold = int(threshold)
            X,y = X[:threshold],y[:threshold]     
        else:
            print(f"[-] The limit {lower_limit} is not in the required interval [0.0:1.0]")
            raise TypeError
    return X,y