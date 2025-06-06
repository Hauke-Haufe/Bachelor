import open3d as o3d
from pathlib import Path
from scene_fragmenter import loop_closure
import json
from config import INTRINSICS_PATH
import os
import multiprocessing
import time

def func():

    with open("config.json", "rb") as file:
            config = json.load(file)
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(os.path.join(INTRINSICS_PATH, "intrinsics.json"))
    intrinsics_matrix = o3d.core.Tensor(intrinsics.intrinsic_matrix)


    t = loop_closure( multiprocessing.Manager().Lock())
    t.run_system(1, 0, 200, config, intrinsics_matrix, "data/images")


s = time.time()
func()
print(time.time()-s)