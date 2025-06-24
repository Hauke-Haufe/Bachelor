import cv2
import pyrealsense2 as rs
import open3d as o3d
import os 
import numpy as np
 
def extract_rs(path, out_path =  "data/extracted" ):

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

                    cv2.imwrite(out_path + "/" + f"frame{i}.png", bgr_image)
                    
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

    freq = 15

    sub_dirs = [d for d in os.listdir(path)]

    i = 0
    for directory in sub_dirs:
        run_path = os.path.join(out_path, f"run{i}")
        extract_images(path + "/" +directory,  run_path,  freq)
        i += 1

def extract_mp4(video_path , run, freq, out_path= "data/extracted"):

    os.makedirs(os.path.join(out_path, run), exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    # Extract frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(os.path.join(out_path, run), f'frame_{frame_count:05d}.png')

        if frame_count % freq == 0:
            frame = cv2.resize(frame, (640, 480))
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    
    

if __name__ == "__main__":  
    #extract_rs("C:/Users/Haufe/Desktop/beachlor/code/data/raw_data/RS/VGA", "C:/Users/Haufe/Desktop/beachlor/code/data/extracted")
    extract_mp4("VIDEO-2025-04-30-13-30-34 2025-05-22 12_22_17.mp4", "run6", 15)
    