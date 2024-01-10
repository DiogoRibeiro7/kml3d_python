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



import numpy as np
import pandas as pd

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
