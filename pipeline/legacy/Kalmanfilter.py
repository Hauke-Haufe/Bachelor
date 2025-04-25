import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class IMUCalmanFilter:

    def __init__(self, dt =0.01):

        self.dt = dt
        self.state = np.zeros(12)
        self.P = np.eye(12) * 0.01

        self.Q = np.eye(12)
        self.Q[:3,:3] *= 0.01
        self.Q[3:6,3:6] *=0.1
        self.Q[6:9,6:9] *= 0.01
        self.Q[9:,9: ] *= 0.001

        self.g = np.array([0,-8.9, 0])
        self.R = np.eye(3) *0.2
        self.I = np.eye(12)

    def set_dt(self, dt):

        self.F[:3, 3:6] = np.eye(3)*dt
        self.B[3:6, :3] = np.eye(3)*dt
        self.B[6:9, 3:6] = np.eye(3)*dt
    
    def set_bias(self, bias):

        self.bias = bias.reshape(6,1)
    
    def predict(self, accel, gyro):
        rot_vec = self.state[0:3]
        vel = self.state[3:6]
        pos = self.state[6:9]
        bias = self.state[9:12]

        omega = gyro -bias
        rot = R.from_rotvec(rot_vec)
        delta_rot = R.from_rotvec(omega* self.dt)
        new_rot = rot*delta_rot
        new_rot_vec = new_rot.as_rotvec()

        acc_world = new_rot.apply(accel)
        acc_corrected = acc_world +self.g

        new_vel = vel +acc_corrected *self.dt
        new_pos = pos + vel*self.dt +0.5 *acc_corrected *self.dt**2

        self.state[0:3] = new_rot_vec
        self.state[3:6] = new_vel
        self.state[6:9] = new_pos

        F = np.eye(12)
        F[3:6, 0:3] = self._skew(acc_world) * -self.dt
        F[6:9, 3:6] = np.eye(3) * self.dt
        F[0:3, 9:12] = -np.eye(3) *self.dt 

        self.P = F@ self.P @ F.T +self.Q

    def update(self,accel):

        rot = R.from_rotvec(self.state[0:3])
        g_est = rot.apply(self.g)
        
        z = accel / np.linalg.norm(accel)
        h = g_est / np.linalg.norm(g_est)
        y = z -h

        H = np.zeros((3,12))
        H[:, 0:3] = self._jacobian_gravity(rot)

        S = H @ self.P @ H.T +self.R
        K = self.P @ H.T@ np.linalg.inv(S)

        self.state += K @ y

        self.P = (self.I - K@ H) @ self.P
    
    def get_pose_matrix(self):
        T = np.eye(4)
        T[:3,:3] = R.from_rotvec(self.state[0:3]).as_matrix()
        T[:3,3] = self.state[6:9]
        return T

    def _skew(self,v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1],v[0], 0]
        ])
    
    def _jacobian_gravity(self, rot):
        eps = 1e-5
        J = np.zeros((3,3))
        for i in range(3):
            dr = np.zeros(3)
            dr[i] = eps
            plus = R.from_rotvec(self.state[0:3] +dr).apply(self.g)
            minus = R.from_rotvec(self.state[0:3] -dr).apply(self.g)
            J[:, i] = (plus -minus)/ (2*eps)

        return J

def test_filter():

    acc_data = np.load("data/images/acceleration.npy")
    gyro_data = np.load("data/images/gyrodata.npy")

    Filter = IMUCalmanFilter(dt = 0.01)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    for i in range(0,200):
        gyro = gyro_data[i][1:]
        accel = acc_data[i][1:]

        Filter.predict(accel, gyro)
        Filter.update(accel)
        pose = Filter.get_pose_matrix()
        pos= pose[:,3]
        direction = pose[:3,:3] @ np.array([0,0,1])
        direction = direction / np.linalg.norm(direction)
        ax.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2])

    plt.show()

if __name__ == "__main__":
    test_filter()