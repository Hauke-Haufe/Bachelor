import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import json
import os
import cv2
from pathlib import Path

#mit dem capture script sollte hiwer alles überarbeitet werden

#das realsense module compeliert nicht unter windows 
def unpack_bag(path, config):

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

def timeframe_imu(gyro_data, acc_data, index, timestamp):
    
    timeframe_gyrodata = []
    timeframe_accdata = []
    while gyro_data[index][0] < timestamp and index < acc_data.shape[0]-1:
        timeframe_gyrodata.append(gyro_data[index][1:3])
        timeframe_accdata.append(acc_data[index][1:3])
        index += 1

    timeframe_accdata = np.asanyarray(timeframe_accdata)
    timeframe_gyrodata = np.asanyarray(timeframe_gyrodata)

    return np.array([timeframe_gyrodata, timeframe_accdata]), index


def unpack_bag_imu(bag_path, config):

    gyro_path = Path(bag_path).parent / "gyrodata.npy"
    acc_path = Path(bag_path).parent / "acceleration.npy"

    reader = o3d.t.io.RSBagReader()
    reader.open(bag_path)
    stream_lenght = reader.metadata.stream_length_usec /1000000
    fps = reader.metadata.fps
    o3d.io.write_pinhole_camera_intrinsic("data/intrinsics.json", reader.metadata.intrinsics)
    reader.close()

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device_from_file(bag_path, repeat_playback=False)

    align = rs.align(rs.stream.color)
    timestamps = []

    acc_data = np.load(acc_path)
    gyro_data = np.load(gyro_path)
    
    freq = int(fps /config["fps"])
    profile = pipeline.start(rs_config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)


    try:
        i = 0 
        passed_time = 0
        index = 0

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

                data, index =timeframe_imu(gyro_data, acc_data, index, timestamp)
                np.save(f"data/images/imu/{int(i/freq)}.npy", data)

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

def clear_dirs():
    paths = ["data/images/color", "data/images/depth", "data/images/imu"]

    for path in paths:
        for file in os.listdir(path):
            file_path = os.path.join(path,file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
def main(path):
    with open("config.json", "rb") as file:
            config = json.load(file)

    clear_dirs()
    unpack_bag(path, config)

if __name__ == "__main__":
    main("data/raw_data/RS/HD/20250414_122808/recording.bag")

