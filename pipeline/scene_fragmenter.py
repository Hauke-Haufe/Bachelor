import open3d as o3d
import numpy as np
import multiprocessing

import json
import time
import os 
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.utils.dlpack

from posegraph import Posegraph
from sensor_io import Framestreamer
from config import FRAGMENT_PATH
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

def is_large_motion(transform, t_threshold, r_threshold):

    big = False
    t = transform[:3,3]
    translation_norm = np.linalg.norm(t)

    
    R_mat = transform[:3,:3]
    rot = R.from_matrix(R_mat)
    angle_rad = rot.magnitude()
    rotation_deg = np.degrees(angle_rad)

    if rotation_deg > r_threshold:
        big = True
        print("big rotation")
    
    if translation_norm> t_threshold:
        big = True
        print("big translation")
    

    return big

class model_tracking:

    def run_system(self, fragment_id, sid, eid, config, intrinsics, path, model = None):

        device = o3d.core.Device("CUDA:0")
        T_frame_model = o3d.core.Tensor(np.identity(4))
        model = o3d.t.pipelines.slam.Model(config["voxel_size"], 16,  10000, T_frame_model, device)

        depth_ref =  depth_image = o3d.t.io.read_image(f"{path}/depth/image{sid}.png")
        input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsics, device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,depth_ref.columns, intrinsics, device)
        poses = []

        for i in range(sid,eid):

            start = time.time()
            color_image = o3d.t.io.read_image(f"{path}/color/image{i}.png")
            depth_image = o3d.t.io.read_image(f"{path}/depth/image{i}.png")

            input_frame.set_data_from_image('depth', depth_image)
            input_frame.set_data_from_image('color', color_image)


            if i > sid:
                try:
                    result = model.track_frame_to_model(input_frame, raycast_frame)
                    T_frame_model = T_frame_model @ result.transformation

                except Exception:
                    print("tracking failed")

            poses.append(T_frame_model.cpu().numpy())
            model.update_frame_pose(i -sid, T_frame_model)
            model.integrate(input_frame)
            model.synthesize_model_frame(raycast_frame)
            print(time.time()-start)

        
        mesh = model.extract_pointcloud()
        #mesh = mesh.transform(T_frame_model)
        o3d.t.io.write_point_cloud(os.path.join(FRAGMENT_PATH, f"{fragment_id}.pcd"), mesh)

class loop_closure:

    #achtung nicht threaded getestet
    def __init__(self, lock):
        self._lock = lock

        self.tl_flag = False
        self.d_flag = False

        self.tl_frame = 0
        self.n_sid = 0
        

    @staticmethod
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

    def create_posegraph_(self, sid, eid,  config, instrinsics, images):

        pose_graph = Posegraph(np.identity(4), config["posegraph_backend"], config["imu"])
        odometry = np.identity(4)

        for source_id in range(sid, eid):
            
            source_image ,_, _,_ = images[source_id]
            images.step_frame()

            for target_id in range(source_id +1, eid , config["key_framefreq"]):
                
                if target_id-source_id < max(2,config["key_framefreq"] * config["num_keyframes"]):
                    
                    target_image, target_accel, target_gyro, tc= images[target_id]

                    if target_id == source_id +1:
                        loop_closure = False
                    else:
                        loop_closure = True

                    success, icp, info= self.odometry_(
                            source_image, target_image, instrinsics.cpu())
                    
                    if target_id == source_id +1 and not success:
                        self.tl_flag = True
                        self.d_flag = True
                        self.tl_frame = source_id

                        return pose_graph

                    if success: 
                        trans = icp.transformation

                        if target_id == source_id +1:
                            odometry = np.dot(trans.numpy(), odometry)
                            pose_graph.add_note(target_id-sid , np.linalg.inv(odometry))
                        if not loop_closure and config["imu"]:
                            pose_graph.add_imu_edge(target_accel, target_gyro, source_id -sid, target_id -sid)

                        pose_graph.add_odometry_edge(trans.numpy(), info, source_id - sid, target_id -sid, loop_closure)

        self.tl_flag = False
        return pose_graph

    @staticmethod
    def integrate_(path, sid,  pose_graph, intrinsics, model = None):

        if model is None:
            vgb = o3d.t.geometry.VoxelBlockGrid(
                attr_names = ('tsdf', 'weight', 'color'),
                attr_dtypes = (o3d.core.Dtype.Float32, o3d.core.Dtype.Float32,o3d.core.Dtype.Float32),
                attr_channels = ((1), (1), (3)),
                voxel_size = 0.01,
                block_resolution = 16,
                block_count = 9000,
                device = o3d.core.Device("CUDA:0")
            )
        else:
            vgb = o3d.t.geometry.VoxelBlockGrid(
                attr_names = ('tsdf', 'weight', 'color', 'label'),
                attr_dtypes = (o3d.core.Dtype.Float32, o3d.core.Dtype.Float32,o3d.core.Dtype.Float32, o3d.core.Dtype.Uint8),
                attr_channels = ((1), (1), (3), (1)),
                voxel_size = 0.01,
                block_resolution = 16,
                block_count = 9000,
                device = o3d.core.Device("CUDA:0")
            )

        p_pose = np.eye(4)

        for i in range(len(pose_graph.nodes)):

            depth_image = o3d.t.io.read_image(f"{path}/depth/image{sid + i}.png").cuda()
            color_image = o3d.t.io.read_image(f"{path}/color/image{sid + i}.png").cuda()
            pose = pose_graph.nodes[i].pose

            frustum_block_coords = vgb.compute_unique_block_coordinates(
                depth_image,  intrinsics, 
                np.linalg.inv(pose)
            )
            
            if model is None:
                vgb.integrate(frustum_block_coords, depth_image, color_image,
                    intrinsics, 
                    np.linalg.inv(pose)
                )
            else:
                image = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(color_image.as_tensor()))
                mask = model(image)
                
                vgb.integrate(frustum_block_coords, depth_image, color_image,
                    mask, intrinsics, intrinsics, 
                    np.linalg.inv(pose))
            
            p_pose =pose

        return vgb

    def run_system(self,fragment_id, sid, eid,  config, intrinsics, path, model = None):
        
        self.n_sid = sid
        image_loader = Framestreamer(Path(path), sid, eid, config)
        pose_graph = self.create_posegraph_(sid, eid,  config, intrinsics, image_loader)

        while self.tl_flag:

            print(f"tracking lost at {self.tl_frame}")

            if pose_graph.count_nodes() > 10:
                pose_graph_opt = pose_graph.optimize()
                vgb = self.integrate_(path, self.n_sid ,pose_graph_opt, intrinsics, model)
                pointcloud = vgb.extract_point_cloud()
                o3d.t.io.write_point_cloud(os.path.join(config["fragment_path"], f"{fragment_id}_{self.n_sid}.pcd"), pointcloud)
                o3d.io.write_pose_graph(os.path.join(config["fragment_path"], f"{fragment_id}_{self.n_sid}.json"), pose_graph_opt)

            
            self.n_sid = sid + self.tl_frame +1
            pose_graph = self.create_posegraph_(self.n_sid, eid, config, intrinsics, path, image_loader)
                
        pose_graph_opt = pose_graph.optimize()
        o3d.io.write_pose_graph(os.path.join(config["fragment_path"], f"{fragment_id}.json"), pose_graph_opt)

        with self._lock:
            if len(pose_graph_opt.nodes) >10:
                vgb = self.integrate_(path, self.n_sid, pose_graph_opt, intrinsics, model)
                pointcloud = vgb.extract_point_cloud()
                o3d.t.io.write_point_cloud(os.path.join(config["fragment_path"], f"{fragment_id}.pcd"), pointcloud)

class Scene_fragmenter:

    @staticmethod
    def load_model_():  

        pass

    def __init__(self, config):
        
        self.config =config

        if config["semantic"]:
            self.model = self.load_model_()
        else:
            self.model = None

        if config["fragmenter_backend"] == "model_tracking": 
            self.backend = model_tracking()

        elif config["fragmenter_backend"] == "loop_closure":
            self.lock = multiprocessing.Manager().Lock()
            self.backend = loop_closure(self.lock)

        else:
            raise Exception("No Valid backend")

    
    def _prepare_task(self):

        for file in os.listdir(self.config["fragment_path"]):
            file_path = os.path.join(self.config["fragment_path"],file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        intrinsics = o3d.io.read_pinhole_camera_intrinsic(os.path.join(self.config["intrinsic_path"], "intrinsics.json"))

        num_images_c = len([file for file in os.listdir(os.path.join(self.config["image_path"], "color")) if file.endswith(".png")])
        num_images_d = len([file for file in os.listdir(os.path.join(self.config["image_path"], "depth")) if file.endswith(".png")])
        num_images = min(num_images_c, num_images_d, config["max_images"])    

        ids = []
        sid, eid = 0,  self.config["frames_per_frag"]
        ids.append([sid, eid])

        while eid-  self.config["frag_overlap"] +self.config["frames_per_frag"]   < num_images:
            sid = eid - self.config["frag_overlap"]
            eid = sid + self.config["frames_per_frag"]
            ids.append([sid, eid])
            
        n_fragments = len(ids)
        print(intrinsics)

        return ids, n_fragments, intrinsics
    
    def make_fragments(self):

        ids, n_fragments, intrinsics = self._prepare_task()

        #achtung hier kann gern mal gpu Ueberfordert werden
        max_workers = 2 #min(max(1, multiprocessing.cpu_count()-1), n_fragments)
        os.environ["OMP_NUM_THREADS"] = '1'
        mp_context = multiprocessing.get_context('spawn')
        intrinsics_matrix = o3d.core.Tensor(intrinsics.intrinsic_matrix)

        if self.config["parallel_fragments"]:
            if self.model == None:
                    
                with mp_context.Pool(processes=max_workers) as pool:
                    args = [(fragment_id, 
                            ids[fragment_id][0], 
                            ids[fragment_id][1], config, 
                            intrinsics_matrix,
                            self.config["image_path"]) for fragment_id in range(n_fragments)]
                    pool.starmap(self.backend.run_system, args)
            
            else:
                mp.set_start_method('spawn', force = True)
                self.model.share_memory()

                processes = []
                for fragment_id in range(n_fragments):
                    p = mp.Process(target = self.backend.run_system, 
                                   args = (fragment_id,
                                            ids[fragment_id][0], 
                                            ids[fragment_id][1], 
                                            config, 
                                            intrinsics_matrix,
                                            self.config["image_path"],
                                            self.model))
                for p in processes:
                    p.join()

        else:

            for fragment_id in range(n_fragments):
            
                self.backend.run_system(fragment_id, 
                                        ids[fragment_id][0], 
                                        ids[fragment_id][1], 
                                        config.copy(), 
                                        intrinsics_matrix, 
                                        self.config["image_path"], self.model)

if __name__ == "__main__":

    with open("config.json", "rb") as file:
        config = json.load(file)

    intrinsics = o3d.io.read_pinhole_camera_intrinsic("data/intrinsics/intrinsics.json")
    intrinsics_matrix = o3d.core.Tensor(intrinsics.intrinsic_matrix)

    p = o3d.io.read_pose_graph("data/test/graph.json")
    l = loop_closure(multiprocessing.Manager().Lock())
    vgb = l.integrate_("data/test",1744, p, intrinsics_matrix)
    pointcloud = vgb.extract_point_cloud()
    o3d.visualization.draw([pointcloud])


    '''start = time.time()
    odo =  Scene_fragmenter(config)
    odo.make_fragments()
    print(time.time()-start)

    pcd = []
    for file in os.listdir("data/fragments"):
        if file.endswith(".pcd"):
            pcd.append(o3d.io.read_point_cloud(os.path.join("data/fragments", file)))
       
    o3d.visualization.draw(pcd[0])'''
    
