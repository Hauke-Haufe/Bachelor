import gtsam
import gtsam.noiseModel
import open3d as o3d
import numpy as np
import json
from pathlib import Path
from typing import Optional
from gtsam import symbol
import time
import pickle

class imuNoise:

    #the data is assument to come form a stationary recording
    def __init__(self):

        self.param = gtsam.PreintegrationCombinedParams(np.array([0,-9.81,0]))
        self.bias = gtsam.imuBias.ConstantBias()

    def from_data(self, g_path, a_path):

        self.accel_data = np.load(a_path)
        self.gyro_data = np.load(g_path)
        self._calc_covs()

    def from_cache(self, cache_path):
        
        with open(cache_path, "r") as f:
            file = json.load(f)
        
        self.param.setGyroscopeCovariance(1*np.array(file["GyroscopeCov"]))
        self.param.setAccelerometerCovariance(1*np.array(file["AccelerometerCov"]))
        self.param.setBiasOmegaCovariance(1 *np.array(file["BiasOmegaCov"]))
        self.param.setBiasAccCovariance(1*np.array(file["BiasAccCov"])) 

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

#assumes the transforms are all form one initial Koordinatesystem
#fuses overlaping graphs with transforms between them

#ACHTUNG möglicher one off fehler
#hier stimm noch garnichts man muss von i die connection auf den i-1 übertragen
def combine_gtsam_posegraphs(graphs, transforms, informations, overlap):

    combined_graph = gtsam.NonLinearFactorGraph()
    combined_initials = gtsam.Values()

    for i in range(1, len(graphs)):

        noise = np.linalg.inv(informations[i] + 1e-6*np.eye(6))
        odom_noise = gtsam.noiseModel.Gaussian.Covariance(noise)
        robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber(1.0), odom_noise)
        graph = graphs[i]

        for j in range(len(graph)):
            
            if j >= len(graph) - overlap:
                combined_graph.add(gtsam.BetweenFactorPose3(
                    symbol(graphs[i].pos_hash, j), 
                    symbol(graphs[i+1].pos_hash,i-len(graph)-overlap)), 
                    gtsam.Pose3(np.eye(4)), robust_noise)
             
            pose = graph.initial.atPose3(symbol(graph.pos_hash, i)).matrix()
            pose = pose @ transforms[i]
            combined_initials.insert(symbol(graph.pos_hash, i), gtsam.Pose3(pose))

            if graph.imu and i > 1:
                combined_initials.insert(
                    symbol(graph.bias_hash, i), 
                    graph.initial.atVector(symbol(graph.bias_hash, i)))
                combined_initials.insert(
                    symbol(graph.vel_hash, i), 
                    graph.initial.atVector((symbol(graph.vel_hash, i))))
        
        for i in range(graph.graph.size()):
            factor =graph.graph.at(i)
            if isinstance(factor, gtsam.BetweenFactorPose3) or isinstance(factor, gtsam.CombinedImuFactor)
                combined_graph.add(factor)

#wrapper class für gtsam
class GTSAMPosegraph:

    #contructs a gtsam Nonlinear Graph and adds a prior
    def __init__(self, intial_pose, sid, imu: Optional[imuNoise] = None):
        
        self.result = None
        self.imu = imu

        self.sid
        self.pos_hash = f'{sid}x'
        self.vel_hash = f'{sid}v'
        self.bias_hash = f'{sid}b'

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.len = 1

        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001]*6))
        self.vel_noise = gtsam.noiseModel.Isotropic.Sigma(3,0.1)
        self.bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 10)
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05,0.05,0.05,0.02, 0.02, 0.02]))
        
        pose0 = gtsam.Pose3(intial_pose)
        self.graph.add(gtsam.PriorFactorPose3(symbol(self.pos_hash, 0), pose0, self.pose_noise))
        self.initial.insert(symbol(self.pos_hash, 0), pose0)

        if not imu is None:

            self.imu_factors = []
            self.imu_initials = []

            vel0 = np.zeros(3)
            self.bias = imu.get_bias()
            self.params = imu.get_params()

            self.imu_factors.append(gtsam.PriorFactorVector(symbol(self.vel_hash, 0), vel0, self.vel_noise))
            self.imu_factors.append(gtsam.PriorFactorConstantBias(symbol(self.bias_hash, 0), self.bias, self.bias_noise))
            self.imu_initials.append((symbol(self.vel_hash, 0), vel0))
            self.imu_initials.append((symbol(self.bias_hash, 0), self.bias))

    def __getitem__(self, key):

        return self.initial.atPose3(symbol(self.pos_hash, key)).matrix()

    def __len__(self):

        return self.len

    def add_odometry_edge(self, odometry,info, i, j, uncertain):
        
        coef = 1
        if uncertain:
            coef = 50

        odom_cov = np.linalg.inv(info + 1e-6*np.eye(6))
        odom_noise = gtsam.noiseModel.Gaussian.Covariance(coef*odom_cov)
        robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber(1.0), odom_noise)
        odometry = gtsam.Pose3(np.linalg.inv(odometry))
        self.graph.add(gtsam.BetweenFactorPose3(symbol(self.pos_hash, i), symbol(self.pos_hash, j), odometry,robust_noise))
    
    def add_note(self, i, trans):
        trans = gtsam.Pose3(trans)

        if not self.initial.exists(symbol(self.pos_hash, i)):
            self.initial.insert(symbol(self.pos_hash, i), trans)
            self.len += 1
        
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

        p_0 = self.initial.atPose3(symbol(self.pos_hash, i)).translation()
        p_1 = self.initial.atPose3(symbol(self.pos_hash, j)).translation()
        dt = gyro[-1][0]- gyro[0][0] 
        v = (p_1 -p_0)/dt 

        self.imu_factors.append(gtsam.CombinedImuFactor(
            symbol(self.pos_hash, i), symbol(self.vel_hash, i),
            symbol(self.pos_hash, j), symbol(self.vel_hash, j),
            symbol(self.bias_hash, i), symbol(self.bias_hash, j), 
            preint))

        self.imu_initials.append((symbol(self.bias_hash, j), self.bias))
        self.imu_initials.append((symbol(self.vel_hash, j), v))
    
    def optimize(self):

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("TERMINATION")
        params.setMaxIterations(500)
        params.setRelativeErrorTol(1e-10)
        params.setAbsoluteErrorTol(1e-10)

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        t = time.time()
        result = optimizer.optimize()
        print(time.time()-t)

        if self.imu:
            for initial in self.imu_initials:
                result.insert(initial[0], initial[1])

            for factor in self.imu_factors:
                self.graph.add(factor)
            
            v = result.atVector(symbol(self.vel_hash, 1))
            result.update(symbol(self.vel_hash, 0), v)

            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, result, params)
            t = time.time()
            result = optimizer.optimize()
            print(time.time()-t)

        self.initial = result
    
    def count_nodes(self):

        def symbolChr(key):
            return (key >> 56) & 0xFF
        len_graph = sum(1 for key in self.graph.keys() if symbolChr(key) ==ord(self.pos_hash))
        
        return len_graph

    #convert values with the coresponding graph to Open3d Posegraph
    def convert_values_to_open3d(self, initial):

        def symbolChr(key):
            return (key >> 56) & 0xFF

        posegraph = o3d.pipelines.registration.PoseGraph()
        len_graph = sum(1 for key in initial.keys() if symbolChr(key) ==ord(self.pos_hash))
        for i in range(len_graph):
            pose = initial.atPose3(symbol(self.pos_hash, i)).matrix()
            node = o3d.pipelines.registration.PoseGraphNode(pose)
            posegraph.nodes.append(node)

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
                posegraph.edges.append(edge)
    
        return posegraph

    def convert_to_open3d(self):

        return self.convert_values_to_open3d(self.initial)

    def save(self, path: str):

        with open(path + ".pkl", "wb") as fb:
            pickle.dump((self.graph, self.initial, self.sid), fb)

    def load(self, path: str):

        with open(path, "rb") as f:
            graph, initial, sid = pickle.load(f)
        
        self.initial = initial
        self.graph = graph

        self.sid = sid
        self.pos_hash = f'{sid}x'
        self.vel_hash = f'{sid}v'
        self.bias_hash = f'{sid}b'

    def get_graph(self):
        return self.graph
    
class Open3dPosegraph():

    def __init__(self, inital_pose = np.eye(4)):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(inital_pose))

    def __getitem__(self, key):
        return self.pose_graph.nodes[key].pose
    
    def __len__(self):
        return len(self.pose_graph.nodes)

    def add_note(self, i, trans):

        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans)) 

    def add_odometry_edge(self, odometry,info, i, j, uncertain):
       
       self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                                i , j ,
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

    def save(self, path: str):

        o3d.io.write_pose_graph(path + ".json", self.pose_graph)

    def load(self, path: str):
        self.pose_graph = o3d.io.read_pose_graph(path)

class Posegraph():

    def __init__(self, backend: str, sid, inital = np.eye(4), imu = False):

        self.imu = imu
        if backend == "gtsam":

            if imu:
                noise = imuNoise()
                noise.from_cache("data/imu_noise.json")
                self.posegraph = GTSAMPosegraph(inital, sid, noise)

            else:
                self.posegraph = GTSAMPosegraph(inital)
        
        elif backend == "Open3D" or backend == "open3d":

            if imu:
                raise RuntimeError("imu integration is not availble for this backend")

            else:
                self.posegraph = Open3dPosegraph(inital)
        
        else:
            raise RuntimeError("no such backend available")
    
    def __getitem__(self, key):

        return self.posegraph[key]
    
    def __len__(self):

        return len(self.posegraph)

    def add_note(self, i, trans):
        self.posegraph.add_note(i, trans)

    def add_odometry_edge(self, odometry,info, i, j, uncertain):
        self.posegraph.add_odometry_edge(odometry, info , i, j, uncertain)
    
    def add_imu_edge(self, accel, gyro, i, j):
        self.posegraph.add_imu_edge(accel, gyro, i, j)

    def optimize(self):
        return self.posegraph.optimize()

    def save(self, filename):
        self.posegraph.save(filename)

    def load(self, filename):
        self.posegraph.load(filename)
    
    def get_graph(self):
        return self.posegraph.get_graph()

if __name__ == "__main__":

    noise = imuNoise()
    noise.from_data("data/test/gyrodata.npy", "data/test/acceleration.npy" )
    noise.save("data")

