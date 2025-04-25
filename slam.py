import open3d as o3d
import numpy as np
from pipeline.reader import load_images_tensor
import json
import time
import os 
import multiprocessing
from config import *

def main():

    with open("config.json", "rb") as file:
        config = json.load(file)

    print(o3d.core.cuda.device_count())

    start = time.time()
    intrinsics = o3d.io.read_pinhole_camera_intrinsic("data/intrinsics.json")  
    intrinsics = o3d.core.Tensor(intrinsics.intrinsic_matrix)

    device = o3d.core.Device("CPU:0")

    T_frame_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(config["voxel_size"], 16,  10000, T_frame_model, device)
    depth_ref =  depth_image = o3d.t.io.read_image(f"data/images/depth/image{0}.png")
    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsics, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,depth_ref.columns, intrinsics, device)
    poses = []

    for i in range(0, config["max_images"]):

        color_image = o3d.t.io.read_image(f"data/images/color/image{i}.png")
        depth_image = o3d.t.io.read_image(f"data/images/depth/image{i}.png")

        input_frame.set_data_from_image('depth', depth_image)
        input_frame.set_data_from_image('color', color_image)

        if i >0:
            result = model.track_frame_to_model(input_frame, raycast_frame)
            T_frame_model = T_frame_model @ result.transformation

        poses.append(T_frame_model.cpu().numpy())
        model.update_frame_pose(i, T_frame_model)
        model.integrate(input_frame)
        model.synthesize_model_frame(raycast_frame)

    print(time.time()-start)
    
    mesh = model.extract_pointcloud()
    o3d.visualization.draw(mesh)

class model_tracking:

    def run_system(self, fragment_id, sid, eid, config, intrinsics, path):

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
                result = model.track_frame_to_model(input_frame, raycast_frame)
                T_frame_model = T_frame_model @ result.transformation

            poses.append(T_frame_model.cpu().numpy())
            model.update_frame_pose(i -sid, T_frame_model)
            model.integrate(input_frame)
            model.synthesize_model_frame(raycast_frame)
            print(time.time()-start)

        
        mesh = model.extract_pointcloud()
        #mesh = mesh.transform(T_frame_model)
        o3d.t.io.write_point_cloud(os.path.join(FRAGMENT_PATH, f"{fragment_id}.pcd"), mesh)


class loop_closure:

    @staticmethod
    def _odometry(source, target, instrinsics):

        try:
            start = time.time()
            result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(source.cuda(),
                                                          target.cuda(),
                                                          instrinsics)
            print(f"odo_time{time.time()-start}")
            success = True
            

            info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(source.depth,
                                                                     target.depth,
                                                                     instrinsics,
                                                                     result.transformation,
                                                                     0.015)   

            return success, result, info.cpu().numpy()

        except Exception as e:
            success = False

            print("Loop closure failed")

            return success, None, None

    @staticmethod 
    def _optimize_posegraph(pose_graph):

        max_correspondence_distance = 0.01
        preference_loop_closure = 0.2

        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
        )
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            preference_loop_closure=preference_loop_closure,
            reference_node=0)

        o3d.pipelines.registration.global_optimization(pose_graph, 
                                                       method, 
                                                       criteria,
                                                       option)
        
        return pose_graph

    @staticmethod
    def _integrate(path, pose_graph, intrinsics):

        vgb = o3d.t.geometry.VoxelBlockGrid(
            attr_names = ('tsdf', 'weight', 'color'),
            attr_dtypes = (o3d.core.Dtype.Float32, o3d.core.Dtype.Float32,o3d.core.Dtype.Float32),
            attr_channels = ((1), (1), (3)),
            voxel_size = 3.0/512,
            block_resolution = 16,
            block_count = 50000,
            #device = o3d.core.Device.CUDA
        )

        for i in range(len(pose_graph.nodes)):

            depth_image = o3d.t.io.read_image(f"{path}/depth/image{i}.png")
            color_image = o3d.t.io.read_image(f"{path}/color/image{i}.png")
            pose = pose_graph.nodes[i].pose

            frustum_block_coords = vgb.compute_unique_block_coordinates(
                depth_image,  intrinsics, 
                np.linalg.inv(pose)
            )

            vgb.integrate(frustum_block_coords, depth_image, color_image,
                intrinsics, 
                np.linalg.inv(pose)
            )
        
        return vgb

    def run_system(self,fragment_id, sid, eid,  config, instrinsics, path):

        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

        images = []
        start = time.time()
        for i in range(sid, eid):
            color_image = o3d.t.io.read_image(f"{path}/color/image{i}.png")
            depth_image = o3d.t.io.read_image(f"{path}/depth/image{i}.png")
            image = o3d.t.geometry.RGBDImage(color_image, depth_image)
            images.append(image)
        print(time.time()-start)

        for source_id in range(sid, eid):
            for target_id in range(source_id +1, eid , config["key_frame_freq"]):
                
                if target_id-source_id <2:

                    if target_id == source_id +1:
                        uncertain = False
                    else:
                        uncertain = True


                    success, icp, info= self._odometry(
                            images[source_id-sid], images[target_id- sid], instrinsics)

                    if success: 
                        trans = icp.transformation
                        odometry = np.dot(trans.numpy(),odometry)
                        pose_graph.nodes.append(
                            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                        )

                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                source_id - sid, target_id -sid,
                                trans.numpy(), info, uncertain
                                )
                        )

        pose_graph = self._optimize_posegraph(pose_graph)
        vgb = self._integrate(path, pose_graph, instrinsics)
        pointcloud = vgb.extract_point_cloud()

        o3d.t.io.write_point_cloud(os.path.join(FRAGMENT_PATH, f"{fragment_id}.pcd"), pointcloud)
        

class odometry_system:

    def __init__(self, backend):
        
        if backend == "model_tracking": 
            self.backend = model_tracking()

        elif backend == "loop_closure":
            self.backend = loop_closure()
    
        else:
            print("None Valid backend")

    #Danger race condition at file loading
    def make_fragments(self, path ,parallel = False):

        for file in os.listdir(FRAGMENT_PATH):
            file_path = os.path.join(FRAGMENT_PATH,file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        with open("config.json", "rb") as file:
            config = json.load(file)
        intrinsics = o3d.io.read_pinhole_camera_intrinsic(os.path.join(INTRINSICS_PATH, "intrinsics.json"))

        num_images_c = len([file for file in os.listdir(os.path.join(path, "color")) if file.endswith(".png")])
        num_images_d = len([file for file in os.listdir(os.path.join(path, "depth")) if file.endswith(".png")])
        num_images = min(num_images_c, num_images_d, config["max_images"])    

        ids = []
        sid, eid, i = 0, config["frames_per_frag"], 0
        ids.append([sid, eid])

        while i < num_images:
            i+= config["frames_per_frag"]
            sid = eid - config["frag_overlap"]
            eid = sid + config["frames_per_frag"]
            ids.append([sid, eid])
            
        n_fragments = len(ids)
        max_workers = min(max(1, multiprocessing.cpu_count()-1), n_fragments)
        os.environ["OMP_NUM_THREADS"] = '1'
        mp_context = multiprocessing.get_context('spawn')

        if parallel:

            with mp_context.Pool(processes=max_workers) as pool:
                args = [(fragment_id, 
                        ids[fragment_id][0], 
                        ids[fragment_id][1], config, 
                        o3d.core.Tensor(intrinsics.intrinsic_matrix),
                        path) for fragment_id in range(n_fragments-1)]
                pool.starmap(self.backend.run_system, args)
        else:

            for fragment_id in range(n_fragments-1):
                self.backend.run_system(fragment_id, 
                                        ids[fragment_id][0], 
                                        ids[fragment_id][1], 
                                        config, 
                                        o3d.core.Tensor(intrinsics.intrinsic_matrix), 
                                        path)
        
   

if __name__ == "__main__":

    odo = odometry_system("model_tracking")
    
    odo.make_fragments("data/images")

    #pcd = o3d.io.read_point_cloud("data/fragments/1.pcd")
    #o3d.visualization.draw([pcd])
    pcd = []
    for file in os.listdir("data/fragments"):
        pcd.append(o3d.io.read_point_cloud(os.path.join("data/fragments", file)))
       
    o3d.visualization.draw(pcd[0])