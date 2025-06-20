import gtsam
import gtsam.noiseModel
import open3d as o3d
import numpy as np
import json
from pathlib import Path
from typing import Optional
from gtsam import symbol
import time

class imuNoise:

    #the data is assument to come form a stationary recording
    def __init__(self):

        self.param = gtsam.PreintegrationCombinedParams(np.array([0,-0.0981,0]))
        self.bias = gtsam.imuBias.ConstantBias()

    def from_data(self, g_path, a_path):

        self.accel_data = 0.01*np.load(a_path)
        self.gyro_data = 0.01* np.load(g_path)
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
        #c = np.array([[1,0,0], [0,0,1], [0,1,0]])
        gyro_bias =np.mean( self.gyro_data[:,1:4] , axis = 0)
        accel_bias = np.mean(self.accel_data[:,1:4], axis = 0) - self.param.n_gravity
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
        return self.bias

#wrapper class für gtsam
class GTSAMPosegraph:

    #contructs a gtsam Nonlinear Graph and adds a prior
    def __init__(self, intial_pose,  imu: Optional[imuNoise] = None):
        
        self.result = None
        self.imu = imu

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001]*6))
        self.vel_noise = gtsam.noiseModel.Isotropic.Sigma(3,0.1)
        self.bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.001)
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05,0.05,0.05,0.02, 0.02, 0.02]))
        
        pose0 = gtsam.Pose3(intial_pose)
        self.graph.add(gtsam.PriorFactorPose3(symbol('x', 0), pose0, self.pose_noise))
        self.initial.insert(symbol('x', 0), pose0)

        if not imu is None:
            vel0 = np.zeros(3)
            self.bias = imu.get_bias()
            self.params = imu.get_params()

            self.graph.add(gtsam.PriorFactorVector(symbol('v', 0), vel0, self.vel_noise))
            self.graph.add(gtsam.PriorFactorConstantBias(symbol('b', 0), self.bias, self.bias_noise))
            self.initial.insert(symbol('v', 0), vel0)
            self.initial.insert(symbol('b', 0), self.bias)

    
    def add_odometry_edge(self, odometry,info, i, j, uncertain):
        
        loop_coef = 1
        if uncertain:
            loop_coef = 10

        odom_cov = np.linalg.inv(info + 1e-6*np.eye(6))
        odom_noise = gtsam.noiseModel.Gaussian.Covariance(loop_coef*odom_cov)
        robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber(1.0), odom_noise)
        odometry = gtsam.Pose3(np.linalg.inv(odometry))
        self.graph.add(gtsam.BetweenFactorPose3(symbol('x', i), symbol('x', j), odometry,robust_noise))
    
    def add_note(self, i, trans):
        trans = gtsam.Pose3(trans)

        if not self.initial.exists(symbol('x', i)):
            self.initial.insert(symbol('x', i), trans)
        
    def add_imu_edge(self, accel, gyro, i, j):

        if self.imu is None:
            raise RuntimeError("No imu config")

        prev = gyro[0][0]
        preint = gtsam.PreintegratedCombinedMeasurements(self.params, self.bias)
        for k in range(1, len(gyro)):
            gyro_data = gyro[k][1:4]
            accel_data = accel[k][1:4]

            dt =  gyro[k][0] -prev
            if dt > 0:
                preint.integrateMeasurement(accel_data,gyro_data, dt)
                prev = gyro[k][0]

        p_0 = self.initial.atPose3(symbol('x', i)).translation()
        p_1 = self.initial.atPose3(symbol('x', j)).translation()
        dt = gyro[len(gyro)-1][0]- gyro[0][0] 
        v = (p_1 -p_0)/dt

        self.graph.add(gtsam.CombinedImuFactor(
            symbol('x', i), symbol('v', i),
            symbol('x', j), symbol('v', j),
            symbol('b', i), symbol('b', j), 
            preint))

        self.initial.insert(symbol('b', j), self.bias)
        self.initial.insert(symbol('v', j), v)
    
    def optimize(self):

        def symbolChr(key):
            return (key >> 56) & 0xFF

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("TERMINATION")
        params.setMaxIterations(500)
        params.setRelativeErrorTol(1e-10)
        params.setAbsoluteErrorTol(1e-10)
        params.setlambdaInitial(1)

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        t = time.time()
        result = optimizer.optimize()
        print(time.time()-t)

        optimsied_posegraph = o3d.pipelines.registration.PoseGraph()
        len_graph = sum(1 for key in result.keys() if symbolChr(key) ==ord('x'))
        for i in range(len_graph):
            pose = result.atPose3(symbol('x', i)).matrix()
            node = o3d.pipelines.registration.PoseGraphNode(pose)
            optimsied_posegraph.nodes.append(node)

        #for debuging
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            if isinstance(factor, gtsam.BetweenFactorPose3):
                keys = factor.keys()
                key1, key2 = gtsam.Symbol(keys[0]),  gtsam.Symbol(keys[1])
                index1, index2 = key1.index(), key2.index()
                noise = np.eye(6)#factor.noiseModel().R()
                odometry = factor.measured().matrix()
                info = np.linalg.inv(noise)

                if abs(index1 - index2) ==1:
                    uncertain = False
                else:
                    uncertain = True
                
                edge = o3d.pipelines.registration.PoseGraphEdge(
                                index1, index2,
                                odometry, info, uncertain
                                )
                optimsied_posegraph.edges.append(edge)


        return optimsied_posegraph
    
    def count_nodes(self):

        def symbolChr(key):
            return (key >> 56) & 0xFF
        len_graph = sum(1 for key in self.graph.keys() if symbolChr(key) ==ord('x'))
        
        return len_graph

    def convert_to_open3d(self):

        def symbolChr(key):
            return (key >> 56) & 0xFF

        optimsied_posegraph = o3d.pipelines.registration.PoseGraph()
        len_graph = sum(1 for key in self.initial.keys() if symbolChr(key) ==ord('x'))
        for i in range(len_graph):
            pose = self.initial.atPose3(symbol('x', i)).matrix()
            node = o3d.pipelines.registration.PoseGraphNode(pose)
            optimsied_posegraph.nodes.append(node)

        #for debuging
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            if isinstance(factor, gtsam.BetweenFactorPose3):
                keys = factor.keys()
                key1, key2 = gtsam.Symbol(keys[0]),  gtsam.Symbol(keys[1])
                index1, index2 = key1.index(), key2.index()
                noise = np.eye(6)#factor.noiseModel().R()
                odometry = factor.measured().matrix()
                info = np.linalg.inv(noise)

                if abs(index1 - index2) ==1:
                    uncertain = False
                else:
                    uncertain = True
                
                edge = o3d.pipelines.registration.PoseGraphEdge(
                                index1, index2,
                                odometry, info, uncertain
                                )
                optimsied_posegraph.edges.append(edge)
    
        return optimsied_posegraph

class Open3dPosegraph():

    def __init__(self, inital_pose):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(inital_pose))

    def add_note(self, i, trans):

        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans)) 

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
        return self.pose_graph

    def count_nodes(self):

        return len(self.pose_graph.nodes)

    def convert_to_open3d(self):

        return self.pose_graph

class Posegraph():

    def __init__(self, inital, backend: str, imu = False):

        self.imu = imu
        if backend == "gtsam":

            if imu:
                noise = imuNoise()
                noise.from_cache("data/imu_noise.json")
                self.posegraph = GTSAMPosegraph(inital, noise)

            else:
                self.posegraph = GTSAMPosegraph(inital)
        
        elif backend == "Open3D" or backend == "open3d":

            if imu:
                raise RuntimeError("imu integration is not availble for this backend")

            else:
                self.posegraph = Open3dPosegraph(inital)
        
        else:
            raise RuntimeError("no such backend available")
    
    def add_note(self, i, trans):
        self.posegraph.add_note(i, trans)

    def add_odometry_edge(self, odometry,info, i, j, uncertain):
        self.posegraph.add_odometry_edge(odometry, info , i, j, uncertain)
    
    def add_imu_edge(self, accel, gyro, i, j):
        self.posegraph.add_imu_edge(accel, gyro, i, j)

    def count_nodes(self):
        return self.posegraph.count_nodes()

    def optimize(self):
        return self.posegraph.optimize()
    
    def convert_to_open3d(self):
        return self.posegraph.convert_to_open3d()


if __name__ == "__main__":

    noise = imuNoise()
    noise.from_data("data/test/gyrodata.npy", "data/test/acceleration.npy" )
    noise.save("data")

