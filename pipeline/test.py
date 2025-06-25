import open3d as o3d
from pathlib import Path
from posegraph import Posegraph
import numpy as np
from scene_fragmenter import loop_closure
from sensor_io import Framestreamer
import json
from config import INTRINSICS_PATH
import os
import multiprocessing
import time

import threading

def create_posegraph_( sid, eid,  config, instrinsics, images):

        pose_graph = Posegraph(np.identity(4), config["posegraph_backend"], config["imu"])
        odometry = np.identity(4)

        for source_id in range(sid, eid):
            
            source_image ,_, _,_ = images[ source_id]
            images.step_frame()

            for target_id in range(source_id +1, eid , config["key_framefreq"]):
                
                if target_id-source_id < max(2,config["key_framefreq"] * config["num_keyframes"]):
                    
                    target_image, target_accel, target_gyro, _= images[target_id]

                    if target_id == source_id +1:
                        loop_closure = False
                    else:
                        loop_closure = True

                    success, icp, info= odometry_(
                            source_image, target_image, instrinsics.cpu())

                    if success: 
                        trans = icp.transformation

                        if target_id == source_id +1:
                            odometry = np.dot(trans.numpy(), odometry)
                            pose_graph.add_note(source_id-sid +1, np.linalg.inv(odometry))
                        if not loop_closure and config["imu"]:
                            pose_graph.add_imu_edge(target_accel, target_gyro, source_id, target_id)

                        pose_graph.add_odometry_edge(trans.numpy(), info, source_id - sid, target_id -sid, loop_closure)

        return pose_graph

def odometry_(source, target, instrinsics):

        try:
            start = time.time()
            critiria = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(
                max_iteration=6, 
                relative_rmse=1.000000e-06, 
                relative_fitness=1.000000e-06), 
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(
                max_iteration=3, 
                relative_rmse=1.000000e-06, 
                relative_fitness=1.000000e-06), 
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(
                max_iteration=1, 
                relative_rmse=1.000000e-06, 
                relative_fitness=1.000000e-06)
            ]
            
            result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                source.cuda(),
                target.cuda(),
                instrinsics, 
                criteria_list = critiria)
            
            #print(f"odo_time:{time.time()-start}")

            info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
                source.depth,
                target.depth,
                instrinsics,
                result.transformation,
                0.015)   

            return True, result, info.cpu().numpy()

        except Exception:
            print("Loop closure failed")

            return False, None, None

def worker(pose_graph, sid, eid,  config, instrinsics, images):
     
   for source_id in range(sid, eid):
            
        source_image ,_, _,_ = images[ source_id]
        images.step_frame()

        for target_id in range(source_id +1, eid , config["key_framefreq"]):
            
            if target_id-source_id < max(2,config["key_framefreq"] * config["num_keyframes"]):
                
                target_image, target_accel, target_gyro, _= images[target_id]

                if target_id == source_id +1:
                    loop_closure = False
                else:
                    loop_closure = True

                success, icp, info= odometry_(
                        source_image, target_image, instrinsics.cpu())

                if success: 
                    trans = icp.transformation

                    if target_id == source_id +1:
                        #odometry = np.dot(trans.numpy(), odometry)
                        #pose_graph.add_note(source_id-sid +1, np.linalg.inv(odometry))
                        pass
                    if not loop_closure and config["imu"]:
                        pass
                        #pose_graph.add_imu_edge(target_accel, target_gyro, source_id, target_id)

                    pose_graph.add_odometry_edge(trans.numpy(), info, source_id - sid, target_id -sid, loop_closure)


def dispatch(config,sid, eid, intrinsics ):
    count = 3
    pose_graph = Posegraph(np.identity(4), config["posegraph_backend"], config["imu"])
    odometry = np.identity(4)

    step = int((eid -sid)/count )
    threads = []
    for i in range(count):
        n_sid = sid + i*step
        n_eid = sid + (i+1)*step
        t = threading.Thread(target=worker, args=
                             (pose_graph, n_sid, n_eid, config, intrinsics,
                              Framestreamer(Path("data/images"), n_sid, n_eid, config)))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()
    
    return pose_graph

def func():

    with open("config.json", "rb") as file:
            config = json.load(file)
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(os.path.join(INTRINSICS_PATH, "intrinsics.json"))
    intrinsics_matrix = o3d.core.Tensor(intrinsics.intrinsic_matrix)
    
    images = Framestreamer(Path("data/images"), 0, 200, config)
    graph1 = create_posegraph_(0,200, config, intrinsics_matrix,  images)


def func2():

    with open("config.json", "rb") as file:
            config = json.load(file)
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(os.path.join(INTRINSICS_PATH, "intrinsics.json"))
    intrinsics_matrix = o3d.core.Tensor(intrinsics.intrinsic_matrix)
    
    graph = dispatch(config, 0, 200, intrinsics_matrix)

s = time.time()
func()
print(time.time()-s)