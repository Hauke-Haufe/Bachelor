import gtsam
import gtsam.imuBias
import gtsam.noiseModel
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
class PoseGraph:

    def __init__(self, imu: Optional[imuNoise] = None):

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3,0.1)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.001)
        odometry = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1,0.1,0.1,]))
        
        pose0 = gtsam.Pose3()
        graph.add(gtsam.PriorFactorpose3(symbol('x', 0), pose0))
        initial.insert(symbol('x', 0), pose0)

        if not imu is None:
            vel0 = np.zeros(3)
            bias0 = imu.get_bias()

            graph.add(gtsam.PriorFactorVector(symbol('v', 0), vel0))
            graph.add(gtsam.PriorfactorConstantBias(symbol('b', 0), bias0))
            initial.ins

        

        if not imu is None:
            gtsam.PreintegrationCombinedMeasurements(imu.get_params(), imu.get_bias())





noise = imuNoise()
noise.from_cache("data/imu_noise.json")
PoseGraph(noise)
