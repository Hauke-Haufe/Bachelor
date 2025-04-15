import pyrealsense2 as rs
import open3d as o3d
from Kalmanfilter import IMUCalmanFilter
from opencv_pose_estimation import pose_estimation
import numpy as np
import json
from reader import load_images_legacy

def load_bag(path, freq):

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(path, repeat_playback=False)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    images = []
    try:
        for i in range(0,100):

            frames = pipeline.wait_for_frames()
            if i% freq== 0:
                frames.keep()

                if not frames.get_depth_frame() or not frames.get_color_frame():
                    continue

                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_image = o3d.geometry.Image(depth_image)
                color_image = o3d.geometry.Image(color_image)
                image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
                    
                images.append(image)

    finally:
        pass

    pipeline.stop()

    return images

def pairwise_odometry(source, target, intrinsics, near,  init_transform = np.identity(4)):
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    option = o3d.pipelines.odometry.OdometryOption(
        depth_diff_max = 0.05,
        depth_max = 3,
        depth_min = 0.4
    )
    #iteration_number_per_pyrimid_level

    if not near:

        success, ode_init = pose_estimation(
            source ,target, intrinsics, False
        )
        
        if success:

            [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source, target, intrinsics, ode_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), 
            option
            )
        
            return [success, trans, info]

        return [False, np.identity(4), np.identity(6)]

    else:
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source, target, intrinsics, init_transform,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), 
            option
        )
    

        return [success, trans, info]

def multiway_odometry(images, sid, eid, key_frame_freq, instrinsics):

    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(trans_odometry)
    )

    for source_id in range(sid, eid):
        for target_id in range(source_id +1, eid):

            if target_id == source_id +1:
                 
                [success, trans, info] = pairwise_odometry(
                    images[source_id], images[target_id], instrinsics, True
                    )
                trans_odometry = np.dot(trans,trans_odometry)

                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans_odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id - sid, target_id -sid,
                        trans, info, uncertain = False
                        )
                )
            
            else:
                if source_id % key_frame_freq == 0 and target_id% key_frame_freq == 0:
                    [success, trans, info] = pairwise_odometry(
                        images[source_id], images[target_id],  
                        instrinsics, False
                    )
            
                    if success:

                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                            source_id - sid, target_id -sid,
                            trans, info, uncertain = True
                            )
                        )
    return pose_graph

def optimize_posegraph(pose_graph):

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

    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria,
                                                   option)
    return pose_graph

def integrade(pose_graph, images, intrinsics):

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length = 0.005,
        sdf_trunc = 0.04,
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.Gray32
    )

    for i in range(len(pose_graph.nodes)):
        
        pose = pose_graph.nodes[i].pose
        volume.integrate(images[i], intrinsics, np.linalg.inv(pose))
    
    mesh = volume.extract_point_cloud()

    return mesh

def main():
    
    path = "data/recording.bag"
    intrinsics = o3d.io.read_pinhole_camera_intrinsic("data/intrinsics.json")

    with open("config.json", "rb") as file:
        config = json.load(file)

    images = load_images_legacy(0, config["max_images"])

    pose_graph = multiway_odometry(images, 0, len(images) , int(len(images)/3) , intrinsics) 
    pose_graph = optimize_posegraph(pose_graph)
    o3d.io.write_pose_graph("data/pose_graph.json", pose_graph)

    mesh = integrade(pose_graph, images, intrinsics)
    o3d.visualization.draw(mesh)

main()