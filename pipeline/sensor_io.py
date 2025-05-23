import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import json
import os
import cv2
from pathlib import Path
from datetime import datetime
import threading


class RSrecorder():
    
    def __init__(self, save_path, HD = False, imu = True):
        
        self.imu = imu
        self.path = Path(save_path)
        
        if HD:
            self.res = (1080, 720)
        else:
            self.res = (640, 480)

        if imu:
            self.gyro_data = []
            self.imuacc_data = []
            self.lock = threading.Lock()

    #beste variante mit Python api
    def stream_imu(self, event):

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
                    
                    with self.lock :
                        if f.get_profile().stream_type() == rs.stream.gyro:
                            self.gyro_data.append([timestamp, data.x, data.y, data.z])
                        else :
                            self.acc_data.append([timestamp, data.x, data.y, data.z])
        
        finally:
            pass

        imu_pipeline.stop()

    def capture(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = self.path / f"{timestamp}.bag"

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.res[0], self.res[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.res[0], self.res[1], rs.format.rgb8, 30)
        
        if self.imu:
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.gyro)

        config.enable_record_to_file(str(bag_path))


        stop_event = threading.Event()
        thread = threading.Thread(target=self.stream_imu, args = (stop_event,))
        thread.start()

        #recording loop
        try:
            while True:
                pass
        
        except KeyboardInterrupt:

            stop_event.set()
            thread.join()
            pipeline.stop()

            gyro_data = np.asanyarray(gyro_data)
            np.save(self.path / "gyrodata.npy", self.gyro_data)

            acc_data = np.asanyarray(acc_data)
            np.save(self.path / "/acceleration.npy", self.acc_data)

    #future set a flag for the recording stop
    #elegantere Lösung
    """
    def capture(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = self.path / f"{timestamp}.bag"

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.res[0], self.res[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.res[0], self.res[1], rs.format.rgb8, 30)
        
        if self.imu:
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.gyro)

        config.enable_record_to_file(str(bag_path))

        #recording loop
        try:
            while True:
                pass
        
        except KeyboardInterrupt:
            pass"""
    
class RSplayer():
    
    def __init__(self, bag_path, config):

        reader = o3d.t.io.RSBagReader()
        reader.open(bag_path)
        self.stream_lenght = reader.metadata.stream_length_usec / 1000000
        self.fps = reader.metadata.fps
        self.intrisics = reader.metadata.intrinsics
        reader.close()

        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device_from_file(bag_path, repeat_playback=False)

        self.align = rs.align(rs.stream.color)
        self.freq = int(self.fps /config["fps"])
        profile = self.pipeline.start(rs_config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

    
    #unpacks the bag into the destination directory
    #funktioniert nicht muss in c++ implementiert werden

    """def unpack(self, config, dest_path):
        
        dest_path = Path(dest_path)
        paths = [dest_path / "color", dest_path / "depth", dest_path / "gyro", dest_path / "accel"]

        for path in paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path,file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        try:
            i = 0 
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            first_timestamp = depth_frame.timestamp / 1000

            timestamp = first_timestamp

            gyro_data = []
            accel_data = []

            while timestamp - first_timestamp < self.stream_lenght and i < config["max_images"]*self.freq: #last one for debug
                frames = self.pipeline.wait_for_frames()

                for f in frames:
                    if f.is_motion_frame():
                        data = f.as_motion_frame.get_motion_data()
                        stream_type = f.get_profile().stream_type()

                        if stream_type == rs.stream.accel:
                            accel_data.append(np.array([data.x, data.y, data.z]))
                        if stream_type == rs.stream.gyro:
                            gyro_data.append(np.array([data.x, data.y, data.z]))

                if i% self.freq == 0:
                    
                    count =int(i/self.freq)

                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not depth_frame or not color_frame:
                        continue
                    
                    timestamp = color_frame.timestamp / 1000

                    depth = np.asanyarray(depth_frame.get_data())
                    cv2.imwrite(dest_path / "depth" / f"image{count}.png", depth.astype(np.uint16))
                    cv2.imwrite(dest_path / "color" / f"image{count}.png", np.asanyarray(color_frame.get_data()))
                    np.save(dest_path / "gyro" / f"{count}.npy", gyro_data)
                    np.save(dest_path / "accel" / f"{count}.npy", accel_data)

                    accel_data = []
                    gyro_data = []
                    
                i += 1

        except Exception as e:
            print(e)

        self.pipeline.stop()"""
    
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

    def unpack(self, bag_path, config):

        gyro_path = Path(bag_path).parent / "gyrodata.npy"
        acc_path = Path(bag_path).parent / "acceleration.npy"

        acc_data = np.load(acc_path)
        gyro_data = np.load(gyro_path)
        
        freq = int(self.fps /config["fps"])

        try:
            i = 0 
            passed_time = 0
            index = 0

            while passed_time < self.stream_lenght-1 and i < config["max_images"]*freq:

                frames = self.pipeline.wait_for_frames()
                
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

        #handles io with file sytem
        def next_frame():
            pass

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
    #main("data/raw_data/RS/HD/20250414_122808/recording.bag")

    with open("config.json", "rb") as file:
        config = json.load(file)

    player = RSplayer("data/recording.bag", config)
    player.unpack(config, "data/images")

