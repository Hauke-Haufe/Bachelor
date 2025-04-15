import open3d as o3d
import open3d.t.pipelines.registration as o3r
from reader import load_point_clouds_from_image, get_transforms
import numpy as np
import json
import time
import os
import gc
import multiprocessing

def global_registration(source, target, source_fpfh, target_fpfh, config):

    start = time.time()
    distance_threshold = config["voxel_size"] * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    print(f"Ransac Time: {time.time()-start}")
    #print(result)
    #o3d.visualization.draw([source.transform(result.transformation), target])

    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, config["voxel_size"] *15 , result.transformation
    )
    return result, information_icp

def multiway_registration(pcds, fpfhs, sid, eid, config, transforms = None):

    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    key_frame_freq = config["key_frame_freq"]


    for source_id in range(sid, eid):
        for target_id in range(source_id +1, eid):

            if target_id == source_id +1:
                

                icp, info= global_registration(
                    pcds[source_id], pcds[target_id], fpfhs[source_id], fpfhs[target_id], config
                    )

                trans = icp.transformation
                odometry = np.dot(trans,odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id - sid, target_id -sid,
                         trans, info, uncertain = False
                        )
                )
            
            else:
                if source_id % key_frame_freq == 0 and target_id% key_frame_freq == 0 and target_id-source_id < 5:
                    icp, info= global_registration(
                    pcds[source_id], pcds[target_id],fpfhs[source_id], fpfhs[target_id],config
                    )
                
                    trans = icp.transformation
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

def sequential_registration(pcds, config, transforms = None):

    merged_pcd = pcds[0]
    icp_transforms = []
    for i in range(1,len(pcds)-1):

        if transforms is None:
            registration, info = pairwise_icp(pcds[i-1], pcds[i], config)
        else:
            registration, info = pairwise_icp(pcds[i-1], pcds[i], config, transforms[i-1])
        

        icp_transforms.append(registration.transformation)


    for i in range(1,len(pcds)-1):
        merged_pcd.transform(icp_transforms[i-1])
        merged_pcd += pcds[i]


    return merged_pcd

def integrade(pose_graph, intrinsics):

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

        depth_image = o3d.t.io.read_image(f"data/images/depth/image{i}.png")
        color_image = o3d.t.io.read_image(f"data/images/color/image{i}.png")
        pose = pose_graph.nodes[i].pose

        frustum_block_coords = vgb.compute_unique_block_coordinates(
            depth_image, o3d.core.Tensor(intrinsics.intrinsic_matrix),  
            np.linalg.inv(pose)
        )

        vgb.integrate(frustum_block_coords, depth_image, color_image,
            o3d.core.Tensor(intrinsics.intrinsic_matrix),  o3d.core.Tensor(intrinsics.intrinsic_matrix), 
            np.linalg.inv(pose)
        )
    
    return vgb

def run():
    start = time.time()
    with open("config.json", "rb") as file:
        config = json.load(file)
    intrinsics = o3d.io.read_pinhole_camera_intrinsic("data/intrinsics.json")

    n_fragments = int(np.ceil(config["max_images"]/config["frames_per_frag"]))
    max_workers = min(max(1, multiprocessing.cpu_count()-1), n_fragments)
    os.environ["OMP_NUM_THREADS"] = '1'
    mp_context = multiprocessing.get_context('spawn')

    with mp_context.Pool(processes=max_workers) as pool:
        args = [(fragment_id, config, intrinsics.intrinsic_matrix) for fragment_id in range(n_fragments-1)]
        pool.starmap(main, args)
    
    print(time.time()-start)

def main(fragment_id, config, intrinsic_matrix):

    sid = (fragment_id) * config['frames_per_frag']
    eid = sid + config['frames_per_frag']
    pcds = load_point_clouds_from_image(intrinsic_matrix, config, sid, eid)

    fpfhs = []
    for i in range(0,len(pcds)):
        pcds[i] = pcds[i].voxel_down_sample(voxel_size = 0.02).to_legacy()
        fgfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcds[i], 
        o3d.geometry.KDTreeSearchParamHybrid(radius= config["voxel_size"] * 6, max_nn = 100)
        )
        fpfhs.append(fgfh)

    pose_graph = multiway_registration(pcds, fpfhs, 0, len(pcds), config)
    pose_graph = optimize_posegraph(pose_graph)
    o3d.io.write_pose_graph(f"data/pose_graphs/{fragment_id}.json", pose_graph)


if __name__ =="__main__": 

    run()
