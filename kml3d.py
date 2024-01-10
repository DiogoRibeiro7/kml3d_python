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
