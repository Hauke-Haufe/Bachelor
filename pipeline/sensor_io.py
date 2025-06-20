import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2

import json
import os
from pathlib import Path
from datetime import datetime
import time
import queue

import threading 
from collections import OrderedDict

class test:
    
    def __init__(self):
        self.color_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        self.gyro_queue = queue.Queue()
        self.accel_queue = queue.Queue()   

    def frame_callback(self, frame):

        stream_type = frame.get_profile().stream_type()
        timestamp = frame.get_timestamp()

        if stream_type == rs.stream.color:
            self.color_queue.put((timestamp, frame))
        elif stream_type == rs.stream.depth:
            self.depth_queue.put((timestamp, frame))
        elif stream_type == rs.stream.gyro:
            self.gyro_queue.put((timestamp, frame))
        elif stream_type == rs.stream.accel:
            self.accel_queue.put((timestamp, frame))

    def monitor(self):

        while True:

            if not self.color_queue.empty():
                ts, _ = self.color_queue.no_wait()
                print(f"color: {ts}")
            if not self.depth_queue.empty():
                ts, _ = self.depth_queue.no_wait()
                print(f"depth: {ts}")
            if not self.gyro_queue.empty():
                ts, _ = self.gyro_queue.no_wait()
                print(f"gyro: {ts}")
            if not self.accel_queue.empty():
                ts, _ = self.accel_queue.no_wait()
                print(f"accel: {ts}")

    def unpack(self, bag_path):

        ctx = rs.context()
        cfg = rs.config()
        cfg.enable_device_from_file(bag_path)

        pipeline = rs.pipeline(ctx)
        profile = pipeline.start(cfg)

        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(True)

        sensors = device.query_sensors()
        for sensor in sensors:
            stream_profiles = sensor.get_stream_profiles()
            sensor.open(stream_profiles)
            sensor.start(self.frame_callback)
        
        thread = threading.Thread(target=self.monitor)
        thread.start()
   
class RSrecorder:
    
    def __init__(self, save_path, HD = False, imu = True):
        
        self.imu = imu
        self.path = Path(save_path)
        
        if HD:
            self.res = (1080, 720)
        else:
            self.res = (640, 480)

        if imu:
            self.gyro_data = []
            self.accel_data = []
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
                            self.accel_data.append([timestamp, data.x, data.y, data.z])

                        
        finally:
            imu_pipeline.stop()

    def capture(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = self.path / f"{timestamp}.bag"

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.res[0], self.res[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.res[0], self.res[1], rs.format.rgb8, 30)
        
        if self.imu:
            stop_event = threading.Event()
            thread = threading.Thread(target=self.stream_imu, args = (stop_event,))
            thread.start()

        config.enable_record_to_file(str(bag_path))
        pipeline.start(config)

        #recording loop
        try:
            while True:
                pass
        
        except KeyboardInterrupt:
            pipeline.stop()
            
            if self.imu:
                stop_event.set()
                thread.join()
                self.gyro_data = np.asanyarray(self.gyro_data)
                np.save(self.path / "gyrodata.npy", self.gyro_data)

                self.accel_data = np.asanyarray(self.accel_data)
                np.save(self.path / "acceleration.npy", self.accel_data)

    #future set a flag for the recording stop
    #elegantere Lösung

    def capture_smart(self):

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
        pipeline.start(config)

        #recording loop
        try:
            while True:
                pass
        
        except KeyboardInterrupt:
            pipeline.stop()
    
class File_io():
    
    def __init__(self, root_dir: str):

        self.root_dir = Path(root_dir)

        self.folders = ["depth", "color", "gyro", "accel"]
        for folder in self.folders:
            if not (self.root_dir / folder).is_dir:
                (self.root_dir / folder).mkdir(parents=True)
        

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
    
    def clean_dirs_(self):

        for path in self.folders:
            if os.path.exists(self.root_dir /path):
                for file in os.listdir(self.root_dir /path):
                    file_path = os.path.join(self.root_dir /path,file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

    @staticmethod
    def timeframe_imu_(gyro_data, acc_data, index, timestamp):
    
        timeframe_gyrodata = []
        timeframe_accdata = []
        while gyro_data[index][0] < timestamp and index < acc_data.shape[0]-1:
            timeframe_gyrodata.append(gyro_data[index][0:4])
            timeframe_accdata.append(acc_data[index][0:4])
            index += 1

        timeframe_accdata = np.asanyarray(timeframe_accdata)
        timeframe_gyrodata = np.asanyarray(timeframe_gyrodata)

        return timeframe_gyrodata, timeframe_accdata, index

    #imu datawith the index i from the i-1 to the i keyframe
    def unpack(self,bag_path, config):
        
        self.clean_dirs_()

        reader = o3d.t.io.RSBagReader()
        reader.open(bag_path)
        self.stream_lenght = reader.metadata.stream_length_usec / 1000000
        self.fps = reader.metadata.fps
        self.intrisics = reader.metadata.intrinsics
        reader.close()

        self.bag_path = bag_path
        self.freq = int(self.fps /config["fps"])
        self.max_images = config["max_images"]

        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device_from_file(bag_path, repeat_playback=False)

        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(rs_config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        gyro_path = Path(self.bag_path).parent / "gyrodata.npy"
        accel_path = Path(self.bag_path).parent / "acceleration.npy"

        if not ( gyro_path.is_file() and accel_path.is_file()):
            raise RuntimeError("no imu data found")
        
        dest_path = Path(self.root_dir)

        accel_raw = np.load(accel_path)
        gyro_raw = np.load(gyro_path)

        try:
            i, index = 0, 0 
            frames = self.pipeline.wait_for_frames()
            timestamp, first_timestamp = frames.timestamp, frames.timestamp

            gyro_data = []
            accel_data = []

            while timestamp - first_timestamp < self.stream_lenght and i < self.max_images*self.freq: #last one for debug
                frames = self.pipeline.wait_for_frames()

                if i% self.freq == 0:
                    
                    count =int(i/self.freq)

                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not depth_frame or not color_frame:
                        continue
                    
                    timestamp = color_frame.timestamp / 1000
                    gyro_data, accel_data, index = self.timeframe_imu_(gyro_raw, accel_raw, index, timestamp)

                    depth = np.asanyarray(depth_frame.get_data())
                    cv2.imwrite(dest_path / "depth" / f"image{count}.png", depth.astype(np.uint16))
                    cv2.imwrite(dest_path / "color" / f"image{count}.png", np.asanyarray(color_frame.get_data()))
                    np.save(dest_path / "gyro" / f"{count}.npy", gyro_data)
                    np.save(dest_path / "accel" / f"{count}.npy", accel_data)

                    accel_data = []
                    gyro_data = []
                    
                i += 1
        finally:
            pass

        self.pipeline.stop()

class Frame_get:

    def __init__(self, root_dir: Path, sid, eid, imu = False):
        images = []
        start = time.time()
        for i in range(sid, eid):
            color_image = o3d.t.io.read_image(root_dir / f"color/image{i}.png")
            depth_image = o3d.t.io.read_image(root_dir / f"depth/image{i}.png")
            image = o3d.t.geometry.RGBDImage(color_image, depth_image)
            images.append(image)
        print(time.time()-start)
        
        self.images = images
        self.sid = sid
    
    def step_frame(self):
        pass

    def __getitem__(self, key):
        return self.images[key], None, None, None

class Frame_server:

    def __init__(self, root_dir: Path, sid, eid, config):
        
        self.imu = config["imu"]
        self.max_size = max(2,config["key_framefreq"] * config["num_keyframes"])

        self.root_dir = root_dir
        dirs = ["color", "accel", "gyro", "depth"]
        
        self.dirs = {}
        for direc in dirs:
            directory = (self.root_dir / direc)
            self.dirs[direc] = directory

        self.buffer = OrderedDict()
        self.loader_thread = threading.Thread(target=self.load_worker)
        self.lock = threading.Lock()

        self.sid = sid
        self.eid = eid
        self.cur = self.sid 


        for i in range(self.max_size):
            image, accel, gyro = self.load_(i)

            with self.lock:
                self.buffer[self.cur] = (image, accel, gyro,  self.cur)
            self.cur += 1

        self.loader_thread.start()
    
    def load_(self, index):

        accel, gyro = None, None
        if self.imu:
            accel = np.load(self.dirs["accel"] / f"{index}.npy") #@ c.T 
            gyro = np.load(self.dirs["gyro"] / f"{index}.npy")# @ c.T

        color_image = o3d.t.io.read_image(self.dirs["color"] / f"image{index}.png")
        depth_image = o3d.t.io.read_image(self.dirs["depth"] / f"image{index}.png")
        image = o3d.t.geometry.RGBDImage(color_image, depth_image).cuda()

        return image, accel, gyro
        
    def load_worker(self):
        
        accel, gyro = None, None
        #c = np.array([[1,0,0,0],[0,1,0,0], [0,0,0,1], [0,0,1,0]])
        while self.cur < self.eid:

            if len(self.buffer) < self.max_size:
                
                image, accel, gyro = self.load_(self.cur)

                with self.lock:
                    self.buffer[self.cur] = (image, accel, gyro,  self.cur)
                
                self.cur += 1

    def step_frame(self):

        with self.lock:
            if len(self.buffer)> 0:
                self.buffer.popitem(last=False)
            

    def __getitem__(self, key):
        
        a = time.time()
        while len(self.buffer) < self.cur- key -10:
            pass
        
        #print(time.time()-a)
        if key in self.buffer:
            with self.lock:
                return self.buffer[key]

        else:
            print("cache wurde verfehlt")
            image, accel, gyro = self.load_(self.cur)

            return image, accel, gyro,  self.cur


if __name__ == "__main__":

    with open("config.json", "rb") as file:
        config = json.load(file)

    #recorder = RSrecorder('data/test')
    #recorder.capture()

    io = File_io("data/images")
    io.unpack("data/raw_data/RS/VGA/20250414_120152/recording.bag", config)

    #t = File_io("data/test")
    #t.unpack("data/test", config)

    