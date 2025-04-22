import open3d as o3d
import numpy as np
from reader import load_images_tensor
import json
import time
import os 
import multiprocessing

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


def slam_multithreaded(fragment_id, config, intrinsics):

    sid = (fragment_id) * config['frames_per_frag']
    eid = sid + config['frames_per_frag']

    device = o3d.core.Device("CUDA:0")

    T_frame_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(config["voxel_size"], 16,  10000, T_frame_model, device)
    depth_ref =  depth_image = o3d.t.io.read_image(f"data/images/depth/image{0}.png")
    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsics, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,depth_ref.columns, intrinsics, device)
    poses = []

    for i in range(sid,eid):

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
    
    mesh = model.extract_pointcloud()
    o3d.t.io.write_PointCloud(f"data/{fragment_id}.obj", mesh)
    #o3d.visualization.draw(mesh)

def run_system():
    with open("config.json", "rb") as file:
        config = json.load(file)
    intrinsics = o3d.io.read_pinhole_camera_intrinsic("data/intrinsics.json")

    n_fragments = int(np.ceil(config["max_images"]/config["frames_per_frag"]))
    max_workers = min(max(1, multiprocessing.cpu_count()-1), n_fragments)
    os.environ["OMP_NUM_THREADS"] = '1'
    mp_context = multiprocessing.get_context('spawn')

    with mp_context.Pool(processes=max_workers) as pool:
        args = [(fragment_id, config, o3d.core.Tensor(intrinsics.intrinsic_matrix)) for fragment_id in range(n_fragments-1)]
        pool.starmap(slam_multithreaded, args)

if __name__ == "__main__":
    run_system()

 