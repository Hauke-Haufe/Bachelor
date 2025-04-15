"""import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from Kalmanfilter import *

pos = np.array([0,0,0]).reshape(3,1)
direction = np.array([0,0,1]).reshape(3,1)

acc_data = np.load("data/images/acceleration.npy")
gyro_data = np.load("data/images/gyrodata.npy")
prev_timestamp = gyro_data[0][0]
positions = []
index = 0

acc_bias = compute_bias_acc(acc_data)
gyro_bias = compute_bias_gyro(gyro_data)
print(acc_bias, gyro_bias)

transforms = np.load("data/transforms.npy")
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

for transform in transforms:

    pos = transform[:,3] + pos
    direction = transform[:3,:3] @ direction
    
    ax.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2])

plt.show()
"""

import open3d as o3d

it = o3d.camera.PinholeCameraIntrinsic(1280, 720, 1067.0000,1067.0699, 1129.5300, 630.8040)
o3d.io.write_pinhole_camera_intrinsic("data/zed_instrinsics.json", it)