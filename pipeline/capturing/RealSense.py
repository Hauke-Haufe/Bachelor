import pyrealsense2 as rs
import threading
import open3d as o3d
import numpy as np
import time
from datetime import datetime
import os
from pathlib import Path
import cv2


gyro_data = []
acc_data = []
lock = threading.Lock()

def record_frames():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.gbr8, 30)
    config.enable_record_to_file("data/recording.bag")

    pipeline.start(config)
    path = "/home/nb-messen-07/Desktop/SpatialMapping/capturing/data/RS/VGA/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path + f"{timestamp}"
    
    os.makedirs( path, exist_ok=True)

    global gyro_data
    global acc_data


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    time.sleep(5)

    pipeline.stop()

def stream_imu(event):

    global gyro_data
    global acc_data

    imu_pipeline = rs.pipeline()
    imu_config = rs.config()
    imu_config.enable_stream(rs.stream.gyro)
    imu_config.enable_stream(rs.stream.accel)

    imu_pipeline.start(imu_config)

    try:
        while not event.is_set():
            frames = imu_pipeline.wait_for_frames()
            for f in frames:
                if f.is_motion_frame():
                    motion = f.as_motion_frame()
                    data = motion.get_motion_data()
                    timestamp = motion.get_timestamp() /1000.0
                
                with lock :
                    if f.get_profile().stream_type() == rs.stream.gyro:
                        gyro_data.append([timestamp, data.x, data.y, data.z])
                    else :
                        acc_data.append([timestamp, data.x, data.y, data.z])
    
    finally:
        pass

    imu_pipeline.stop()

def record_frames_imuHD():
    path = "/home/nb-messen-07/Desktop/SpatialMapping/capturing/data/RS/HD/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path + f"{timestamp}"
    
    os.makedirs( path, exist_ok=True)

    global gyro_data
    global acc_data

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    
    config.enable_record_to_file(path +"/recording.bag")

    stop_event = threading.Event()
    thread = threading.Thread(target=stream_imu, args = (stop_event,))
    thread.start()

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    print("record")
    #depth_sensor.set_option(rs.option.visual_preset, 3)
    #depth_sensor.set_option(rs.option.visual_preset, 1)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        stop_event.set()
        thread.join()
        pipeline.stop()

        gyro_data = np.asanyarray(gyro_data)
        np.save(path +"/gyrodata.npy",gyro_data)

        acc_data = np.asanyarray(acc_data)
        np.save(path +"/acceleration.npy", acc_data)

def record_frames_imu():

    path = "/home/nb-messen-07/Desktop/SpatialMapping/capturing/data/RS/VGA/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path + f"{timestamp}"
    
    os.makedirs( path, exist_ok=True)

    global gyro_data
    global acc_data


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    config.enable_record_to_file(path + "/recording.bag")

    stop_event = threading.Event()
    thread = threading.Thread(target=stream_imu, args = (stop_event,))
    thread.start()

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    #depth_sensor.set_option(rs.option.visual_preset, 3)
    depth_sensor.set_option(rs.option.visual_preset, 1)
    print("recoord")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        stop_event.set()
        thread.join()
        pipeline.stop()

        gyro_data = np.asanyarray(gyro_data)
        np.save(path +"/gyrodata.npy",gyro_data)

        acc_data = np.asanyarray(acc_data)
        np.save(path + "/acceleration.npy", acc_data)

record_frames_imuHD()
#record_frames_imu()

'''def record_frames_imu():
    global gyro_data
    global acc_data

    path = "capturing/data/RS/VGA/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path + f"{timestamp}/"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    config.enable_record_to_file("recording.bag")

    stop_event = threading.Event()
    thread = threading.Thread(target=stream_imu, args = (stop_event,))
    thread.start()

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    #depth_sensor.set_option(rs.option.visual_preset, 3)
    depth_sensor.set_option(rs.option.visual_preset, 1)

    time.sleep(10)
    stop_event.set()
    thread.join()
    pipeline.stop()

    gyro_data = np.asanyarray(gyro_data)
    np.save("gyrodata.npy",gyro_data)

    acc_data = np.asanyarray(acc_data)
    np.save("acceleration.npy", acc_data)'''