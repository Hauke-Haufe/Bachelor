import cv2
import pyrealsense2 as rs
import open3d as o3d
import os 
import numpy as np
from pathlib import Path

def extract_frames_rsbag(bag_path: Path , out_dir: Path, freq =15):

    reader = o3d.t.io.RSBagReader()
    reader.open(str(bag_path))
    stream_lenght = reader.metadata.stream_length_usec /1000000
    reader.close()

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device_from_file(str(bag_path), repeat_playback=False)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(rs_config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    try:
        i = 0 
        passed_time = 0

        while passed_time < stream_lenght-2:

            frames = pipeline.wait_for_frames()
            
            if i % freq == 0:

                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()

                if not color_frame:
                    continue
                
                timestamp = color_frame.timestamp / 1000
                bgr_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)

                cv2.imwrite(out_dir / f"frame_{i:05d}.png", bgr_image)
                
                if i == 0:
                    prev_timestamp = timestamp
                else:
                    passed_time += timestamp - prev_timestamp
                    prev_timestamp = timestamp
            i += 1

    finally:
        pass

    pipeline.stop()

def extract_mp4(video_path: Path , out_path: Path, freq = 15):

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = out_path / f'frame_{frame_count:05d}.png'

        if frame_count % freq == 0:
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    
    