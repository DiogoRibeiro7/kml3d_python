import pandas as pd
import numpy as np
from scipy.spatial import distance as scipy_distance


class ParKml:
    def __init__(self, saveFreq, maxIt, imputationMethod, distanceName, power, distance, centerMethod, startingCond, nbCriterion, scale):
        self.saveFreq = saveFreq
        self.maxIt = maxIt
        self.imputationMethod = imputationMethod
        self.distanceName = distanceName
        self.power = power
        self.distance = distance
        self.centerMethod = centerMethod
        self.startingCond = startingCond
        self.nbCriterion = nbCriterion
        self.scale = scale


def parKml3d(saveFreq=100, maxIt=200, imputationMethod="copyMean", distanceName="euclidean3d", power=2, centerMethod=np.nanmean, startingCond="nearlyAll", nbCriterion=100, scale=True):
    # Type checks for robustness
    if not isinstance(saveFreq, int):
        raise TypeError("saveFreq must be an integer")
    if not isinstance(maxIt, int):
        raise TypeError("maxIt must be an integer")
    # ... additional type checks for other parameters can be added as needed

    def euclidean_distance(x, y):
        return scipy_distance.euclidean(x, y)

    distance = None
    if distanceName == "euclidean3d":
        distance = euclidean_distance

    return ParKml(saveFreq, maxIt, imputationMethod, distanceName, power, distance, centerMethod, startingCond, nbCriterion, scale)


# Example usage
kml_params = parKml3d()


def calcul_traj_mean_3d(traj, clust, center_method=np.nanmean):
    # Type checks for robustness
    if not isinstance(traj, np.ndarray):
        raise TypeError("traj must be a numpy array")
    if not isinstance(clust, (list, np.ndarray)):
        raise TypeError("clust must be a list or a numpy array")

    # Ensure clust is a numpy array for easier handling
    clust = np.array(clust)

    # Check dimensions
    if traj.ndim != 3:
        raise ValueError("traj must be a 3-dimensional array")

    # Initialize an empty array for the means
    traj_mean = np.empty_like(traj)

    # Iterate over the 2nd and 3rd dimensions
    for i in range(traj.shape[1]):
        for j in range(traj.shape[2]):
            # Extract the slice
            slice_ = traj[:, i, j]

            # Convert to a DataFrame for easy groupby operation
            df = pd.DataFrame({'slice': slice_, 'clust': clust})

            # Group by cluster and apply the center_method
            group_means = df.groupby('clust')['slice'].apply(center_method)

            # Assign the means back to the traj_mean array
            for k, mean in group_means.items():
                traj_mean[clust == k, i, j] = mean

    return traj_mean

# Example usage
# traj = np.random.rand(100, 10, 10)  # Example 3D array
# clust = np.random.randint(0, 5, 100)  # Example cluster assignments
# traj_mean = calcul_traj_mean_3d(traj, clust)


def affect_indiv_3d(traj, clusters_center, distance_func):
    # Type checks for robustness
    if not isinstance(traj, np.ndarray):
        raise TypeError("traj must be a numpy array")
    if not isinstance(clusters_center, np.ndarray):
        raise TypeError("clusters_center must be a numpy array")

    nb_id = traj.shape[0]
    cluster_affectation = np.ones(nb_id, dtype=int)

    # Initial distance calculation
    dist_current = np.array(
        [distance_func(traj[i, :, :], clusters_center[0, :, :]) for i in range(nb_id)])

    # Iterate over each cluster center
    for i_nb_clusters in range(1, clusters_center.shape[0]):
        dist_to_mean = np.array([distance_func(
            traj[i, :, :], clusters_center[i_nb_clusters, :, :]) for i in range(nb_id)])

        # Update the cluster assignment
        cond = dist_to_mean < dist_current
        cluster_affectation[cond] = i_nb_clusters + \
            1  # +1 because Python is zero-indexed
        dist_current = np.where(
            dist_to_mean < dist_current, dist_to_mean, dist_current)

    return cluster_affectation

# Example usage
# Define a distance function, for example, Euclidean distance


def dist3d(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# traj = np.random.rand(100, 10, 10)  # Example 3D array
# clusters_center = np.random.rand(5, 10, 10)  # Example cluster centers
# cluster_affectation = affect_indiv_3d(traj, clusters_center, dist3d)
