import open3d as o3d
import numpy as np
from reader import load_images_tensor
import json

def main():

    with open("config.json", "rb") as file:
        config = json.load(file)

    intrinsics = o3d.io.read_pinhole_camera_intrinsic("data/intrinsics.json")  
    intrinsics = o3d.core.Tensor(intrinsics.intrinsic_matrix)

    device = o3d.core.Device("CUDA:0")

    T_frame_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(config["voxel_size"], 16,  50000)
    depth_ref =  depth_image = o3d.t.io.read_image(f"data/images/depth/image{0}.png")
    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsics, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,depth_ref.columns, intrinsics, device)
    poses = []

    for i in range(config["max_images"]):

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

    o3d.visualization.draw(mesh)

main()