import gtsam
import gtsam.noiseModel
import open3d as o3d
import numpy as np
import json
from pathlib import Path
from typing import Optional
from gtsam import symbol


class imuNoise:

    #the data is assument to come form a stationary recording
    def __init__(self):

        self.param = gtsam.PreintegrationCombinedParams(np.array([0,-9.81,0]))
        self.bias = gtsam.imuBias.ConstantBias()

    def from_data(self, g_path, a_path):

        self.accel_data = np.load(g_path)
        self.gyro_data = np.load(a_path)
        self._calc_covs()

    def from_cache(self, cache_path):
        
        with open(cache_path, "r") as f:
            file = json.load(f)
        
        self.param.setGyroscopeCovariance(np.array(file["GyroscopeCov"]))
        self.param.setAccelerometerCovariance(np.array(file["AccelerometerCov"]))
        self.param.setBiasOmegaCovariance(np.array(file["BiasOmegaCov"]))
        self.param.setBiasAccCovariance(np.array(file["BiasAccCov"])) 

        self.bias = gtsam.imuBias.ConstantBias(np.array(file["BiasAcc"]), np.array(file["BiasOmega"]))

        
    def _calc_cov_accel(self):
        self.param.setAccelerometerCovariance(np.cov(self.accel_data[:,1:4].T))
    
    def _calc_cov_gyro(self):
        self.param.setGyroscopeCovariance(np.cov(self.gyro_data[:,1:4].T))
    
    def _calc_bias_cov_gyro(self):

        n_windows = 100
        len_window = int(len(self.gyro_data) / n_windows)

        gyro_means = []
        for i in range(n_windows):
            gyro_means.append(np.mean(self.gyro_data[i*len_window:(i+1)*len_window,1:4], axis = 0))

        gyro_means = np.asanyarray(gyro_means)
        self.param.setBiasOmegaCovariance(np.cov(gyro_means.T))
    
    def _calc_bias_cov_accel(self):

        n_windows = 100
        len_window = int(len(self.accel_data) / n_windows)

        accel_means = []
        for i in range(n_windows):
            accel_means.append(np.mean(self.accel_data[i*len_window:(i+1)*len_window,1:4], axis = 0))

        accel_means = np.asanyarray(accel_means)
        self.param.setBiasAccCovariance(np.cov(accel_means.T))

    def _calc_bias(self):
        gyro_bias =np.mean( self.gyro_data[:,1:4], axis = 0)- self.param.n_gravity
        accel_bias = np.mean(self.accel_data[:,1:4], axis = 0) 
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        self.bias = bias

    def _calc_covs(self):
        
        self._calc_cov_accel()
        self._calc_cov_gyro()
        self._calc_bias_cov_accel()
        self._calc_bias_cov_gyro()
        self._calc_bias()

    def save(self, save_dir):

        with open(Path(save_dir)/"imu_noise.json", "w") as f:
            json.dump({
                "GyroscopeCov" : self.param.getGyroscopeCovariance().tolist(),
                "AccelerometerCov": self.param.getAccelerometerCovariance().tolist(),
                "BiasOmegaCov": self.param.getBiasOmegaCovariance().tolist(),
                "BiasAccCov": self.param.getBiasAccCovariance().tolist(),
                "BiasOmega": self.bias.accelerometer().tolist(),
                "BiasAcc": self.bias.gyroscope().tolist()
            }, f, indent= 4)

    def get_params(self):
        return self.param
    
    def get_bias(self):
        return self.bias()




#wrapper class für gtsam
class GTSAMPosegraph:

    def __init__(self, intial_pose,  imu: Optional[imuNoise] = None):

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
        self.vel_noise = gtsam.noiseModel.Isotropic.Sigma(3,0.1)
        self.bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.001)
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1,0.1,0.1,]))
        
        pose0 = gtsam.Pose3(intial_pose)
        self.graph.add(gtsam.PriorFactorpose3(symbol('x', 0), pose0), self.pose_noise)
        self.initial.insert(symbol('x', 0), pose0)

        if not imu is None:
            vel0 = np.zeros(3)
            self.bias = imu.get_bias()
            self.params = imu.get_params()

            self.graph.add(gtsam.PriorFactorVector(symbol('v', 0), vel0, self.vel_noise))
            self.graph.add(gtsam.PriorfactorConstantBias(symbol('b', 0), self.bias, self.bias_noise))
            self.initial.insert(symbol('v', 0), vel0)
            self.initial.insert(symbol('b', 0), self.bias)

    
    def add_odometry_edge(self, odometry,info, i, j, uncertain):

        odom_cov = np.linalg.inv(info + 1e-6 *np.eye(6))
        odom_noise = gtsam.noiseModel.Gaussian.Covariance(odom_cov)
        self.graph.add(gtsam.BetweenFactorPose3(symbol('x', i), symbol('x', j), odometry, odom_noise))
    
    def add_note(self, i, trans):

        self.intial.insert(symbol('x', i), trans)

    def add_imu_edge(self, accel, gyro, i, j):

        if self.imu is None:
            raise RuntimeError("No imu config")

        prev = gyro[0][0]
        preint = gtsam.PreintegratedCombinedMeasurements(self.params, self.bias)
        for i in range(len(gyro)):

            dt = prev - gyro[i][0]
            preint.integrate(accel[i][1:4], gyro[i][1:4], dt)
            prev = gyro[i][0]

        self.graph.add(gtsam.CombinedImuFactor(
            symbol('x', i), symbol('v', i),
            symbol('x', j), symbol('v', j),
            symbol('b', i), symbol('b', j), 
            preint))

        self.inital.insert(symbol('b', j), self.bias)
        self.initial.insert(symbol('v', j), )
    
    def optimize(self):

        optimizer = gtsam.LevenbergMarquardOptimizer(self.graph, self.initial)
        result = optimizer.optimise()

class Open3dPosegraph():

    def __init__(self, inital_pose):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(inital_pose))

    def add_note(self, i, trans):

        self.pose_graph[i] = o3d.pipelines.registration.PoseGraphNode(trans)

    def add_odometry_edge(self, odometry,info, i, j, uncertain):
       
       self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                                i, j,
                                odometry, info, uncertain
                                ))

    def optimize(self):
        max_correspondence_distance = 0.01
        preference_loop_closure = 0.2

        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
        )
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            preference_loop_closure=preference_loop_closure,
            reference_node=0)

        o3d.pipelines.registration.global_optimization(self.pose_graph, 
                                                       method, 
                                                       criteria,
                                                       option)

class Posegraph():

    def __init__(self, inital, backend: str, imu = False):

        self.imu = imu
        if backend == "gtsam":

            if imu:
                noise = imuNoise()
                noise.from_cache("data/imu_noise.json")
                self.posegraph = GTSAMPosegraph(inital, noise)

            else:
                self.posegraph = GTSAMPosegraph(inital, noise)
        
        elif backend == "open3d":

            if imu:
                raise RuntimeError("imu integration is not availble for this backend")

            else:
                self.posegraph = Open3dPosegraph(inital)
    
    def add_note(self, i, trans):
        self.backend.add_note(i, trans)

    def add_odometry_edge(self, odometry,info, i, j, uncertain):
        self.backend.add_odometry_edge(odometry, info , i, j, uncertain)
    
    def add_imu_edge(self, accel, gyro, i, j):
        self.backend.add_imu_edge(accel, gyro, i, j)
    def optimize(self):
        self.backend.optimize()

