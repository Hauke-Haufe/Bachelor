import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import json
import os
import cv2
#from Kalmanfilter import IMUCalmanFilter


def unpack_bag(path, config):

    print(os.path.exists(path))

    reader = o3d.t.io.RSBagReader()
    reader.open(path)
    stream_lenght = reader.metadata.stream_length_usec /1000000
    fps = reader.metadata.fps
    o3d.io.write_pinhole_camera_intrinsic("data/intrinsics.json", reader.metadata.intrinsics)
    reader.close()

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device_from_file(path, repeat_playback=False)

    align = rs.align(rs.stream.color)
    timestamps = []

    freq = int(fps /config["fps"])
    profile = pipeline.start(rs_config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)


    try:
        i = 0 
        passed_time = 0

        while passed_time < stream_lenght-1 and i < config["max_images"]*freq:

            frames = pipeline.wait_for_frames()
            
            if i% freq == 0:

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue
                
                timestamp = color_frame.timestamp / 1000
                timestamps.append(timestamp)


                depth = np.asanyarray(depth_frame.get_data())
                cv2.imwrite(f"data/images/depth/image{int(i/freq)}.png", depth.astype(np.uint16))
                cv2.imwrite(f"data/images/color/image{int(i/freq)}.png", np.asanyarray(color_frame.get_data()))

                if i == 0:
                    prev_timestamp = timestamp
                else:
                    passed_time += timestamp - prev_timestamp
                    prev_timestamp = timestamp
                
            i += 1

    finally:
        pass

    pipeline.stop()
    #np.save("data/images/timestamps.npy", np.asanyarray(timestamps))

""""def estimate_transform(gyro_data, acc_data, timestamp, index):

    prev_timestamp = gyro_data[index][0]
    Filter = IMUCalmanFilter(dt = 0.01)
    while gyro_data[index][0] < timestamp and index < acc_data.shape[0]-1:

                    cur_timestamp = gyro_data[index][0]
                    dt = cur_timestamp -prev_timestamp

                    if dt != 0:
                        imu_data = np.concatenate((acc_data[index][1:4], gyro_data[index][1:4])).reshape((6,1))
                        Filter.set_dt(dt)
                        Filter.predict(imu_data)
                        Filter.update(imu_data)
                        
                    index += 1
                    prev_timestamp = cur_timestamp 

    transform = Filter.get_transform()
    return transform , index

def get_transforms():
    
    timestamps = np.load("data/images/timestamps.npy")
    acc_data = np.load("data/images/acceleration.npy")
    gyro_data = np.load("data/images/gyrodata.npy")
    index = 0

    transforms = []
    for timestamp in timestamps:
         
        transform, index = estimate_transform(gyro_data, acc_data, timestamp, index)
        transforms.append(np.linalg.inv(transform))
    
    print("Transforms loaded")
    
    return np.asanyarray(transforms)

def load_images_legacy(sid, eid):

    images = []
    for i in range(sid, eid):
        color_image = o3d.io.read_image(f"data/images/color/image{i}.png")
        depth_image = o3d.io.read_image(f"data/images/depth/image{i}.png")

        image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
        images.append(image)#
    
    return images

def load_images_tensor(sid, eid):

    images = []
    for i in range(sid, eid):
        color_image = o3d.t.io.read_image(f"data/images/color/image{i}.png")
        depth_image = o3d.t.io.read_image(f"data/images/depth/image{i}.png")

        image = o3d.t.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
        images.append(image)
    
    return images

def load_point_clouds_from_image(intrinsics, config, sid, eid):

    num_images = len(os.listdir("data/images/color"))
    pcds = []

    for i in range(sid, eid):

        color_image = o3d.t.io.read_image(f"data/zed_images/color/image{i}.png")
        depth_image = o3d.t.io.read_image(f"data/zed_images/depth/image{i}.png")

        image = o3d.t.geometry.RGBDImage(color_image, depth_image, aligned = True)
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(image, intrinsics, with_normals = True,
            depth_max = config["depth_max"]    
        )
        o3d.visualization
        pcds.append(pcd.cuda()) ############## Achtung

    print("Pointclouds loaded")
    return pcds
"""

def clear_dirs():
    path_1 = "data/images/color"
    path_2 = "data/images/depth"

    for file in os.listdir(path_1):
        file_path = os.path.join(path_1,file)
        if os.path.isfile(file_path):
             os.remove(file_path)
    
    for file in os.listdir(path_2):
        file_path = os.path.join(path_2,file)
        if os.path.isfile(file_path):
             os.remove(file_path)

def main(path):
    with open("config.json", "rb") as file:
            config = json.load(file)

    clear_dirs()
    unpack_bag(path, config)

if __name__ == "__main__":
    main("data/raw_data/RS/HD/20250414_122808/recording.bag")

