import gtsam
import numpy as np

#wrapper class für gtsam
class PoseGraph:

    def __init__(self):

        ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1,0.1,0.1,]))

        graph = gtsam.NonlinearFactorGraph()

    

