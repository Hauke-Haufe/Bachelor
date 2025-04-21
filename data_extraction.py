import cv2
import pyrealsense2 as rs
import open3d as o3d
import os 
import numpy as np
 

def extract_images(path, out_path, freq):

    path = path + "/recording.bag"

    reader = o3d.t.io.RSBagReader()
    reader.open(path)
    stream_lenght = reader.metadata.stream_length_usec /1000000
    fps = reader.metadata.fps
    reader.close()

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device_from_file(path, repeat_playback=False)

    align = rs.align(rs.stream.color)
    timestamps = []

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
                #timestamps.append(timestamp)
                bgr_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)

                cv2.imwrite(out_path + "/" + f"{timestamp}.png", bgr_image)
                
                if i == 0:
                    prev_timestamp = timestamp
                else:
                    passed_time += timestamp - prev_timestamp
                    prev_timestamp = timestamp
                
            i += 1

    finally:
        pass

    pipeline.stop()
    #np.save("data/extracted/timestamps.npy", np.asanyarray(timestamps))

def extract_rs(path):

    freq = 15

    out_path = "C:/Users/Haufe/Desktop/beachlor/code/data/extracted"
    sub_dirs = [d for d in os.listdir(path)]

    for directory in sub_dirs:
        extract_images(path + "/" +directory, out_path,  freq)


extract_rs("C:/Users/Haufe/Desktop/beachlor/code/data/raw_data/RS/VGA")