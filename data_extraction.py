import opencv
import rs
import open3d as o3d
import os 



def extract_rs(path):

    freq = 20
    sub_dirs = [d for d in os.listdir(path) if os.path.isdir(d)]

    for directory in subdir:
        
        extract_images(directory, freq)

