import open3d as o3d
import open3d.t.pipelines.registration as o3r
from reader import load_point_clouds_from_image, get_transforms
import numpy as np
import json
import time
import os
import multiprocessing

def pairwise_registration(source, target, config, transform= None):

    if config["color"] == True:
        estimator = o3r.TransformationEstimationForColoredICP()
    else: 
        estimator = o3r.TransformationEstimationPointToPlane()

    criteria = o3r.ICPConvergenceCriteria(
        relative_fitness = 0.000005, #unterschied in fitness 
        relative_rmse = 0.000005, #unterschied in mse
        max_iteration = config["max_iteration"])

    max_corespondence_distance_coarse = config["voxel_size"] *15 
    max_corespondence_distance_fine = config["voxel_size"] *1.5

    #odometry
    if not transform is None:
        fine_icp = o3r.icp(
            source, target, max_corespondence_distance_fine, 
            transform, estimator, criteria, voxel_size = config["voxel_size"] )

    #loop closure
    else:
        #ersetze mit global Registration
        start = time.time()
        if config["ransac"]:
            coarse_icp = global_registration(source.to_legacy(), target.to_legacy(), config)
        else:
            coarse_icp= o3r.icp(
                source, target, max_corespondence_distance_coarse, 
                np.identity(4), estimator,criteria, voxel_size =config["voxel_size"]*2.5)
        print(f"coarse Time: {time.time()-start}")
        #print(coarse_icp)    
        #o3d.visualization.draw([source.transform(coarse_icp.transformation), target])
        

        start = time.time()
        fine_icp = o3r.icp(
            source, target, max_corespondence_distance_fine,  
            coarse_icp.transformation, estimator, criteria,
            voxel_size = config["voxel_size"])
        print(f"fine Time: {time.time()-start}")
    
    information_icp = o3d.t.pipelines.registration.get_information_matrix(
        source, target, max_corespondence_distance_fine, fine_icp.transformation
    )

    #print(fine_icp.num_iterations)
    #o3d.visualization.draw([source.transform(fine_icp.transformation), target])

    return fine_icp, information_icp

def global_registration(source, target, config):

    start = time.time()
    radius_feature = config["voxel_size"] * 6
    distance_threshold = config["voxel_size"] * 1.5
    source = source.voxel_down_sample(voxel_size = 0.02)
    target = target.voxel_down_sample(voxel_size = 0.02)

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, 
        o3d.geometry.KDTreeSearchParamHybrid(radius= radius_feature, max_nn = 100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, 
        o3d.geometry.KDTreeSearchParamHybrid(radius= radius_feature, max_nn = 100)
    )

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

    #print(f"Ransac Time: {time.time()-start}")
    #print(result)
    #o3d.visualization.draw([source.transform(result.transformation), target])

    return result

def multiway_registration(pcds, sid, eid, config, transforms = None):

    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    key_frame_freq = config["key_frame_freq"]


    for source_id in range(sid, eid):
        for target_id in range(source_id +1, eid):

            if target_id == source_id +1:
                
                if transforms == None:
                    icp, info= pairwise_registration(
                        pcds[source_id], pcds[target_id], config
                        )
                else:
                    icp, info= pairwise_registration(
                        pcds[source_id], pcds[target_id], config, 
                        transforms[source_id]
                        )

                trans = icp.transformation
                odometry = np.dot(trans.numpy(),odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id - sid, target_id -sid,
                         trans.numpy(), info.numpy(), uncertain = False
                        )
                )
            
            else:
                if source_id % key_frame_freq == 0 and target_id% key_frame_freq == 0 and target_id-source_id < 5:
                    icp, info= pairwise_registration(
                    pcds[source_id], pcds[target_id],config
                    )
            
                   
                    if not icp.num_iterations == config['max_iteration']:

                        trans = icp.transformation
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                            source_id - sid, target_id -sid,
                            trans.numpy(), info.numpy(), uncertain = True
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

        depth_image = o3d.t.io.read_image(f"data/zed_images/depth/image{i}.png")
        color_image = o3d.t.io.read_image(f"data/zed_images/color/image{i}.png")
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

def main():

    with open("config.json", "rb") as file:
        config = json.load(file)
    intrinsics = o3d.io.read_pinhole_camera_intrinsic("/home/nb-messen-07/Desktop/SpatialMapping/data/zed_instrinsics.json")

    pcds = load_point_clouds_from_image(intrinsics.intrinsic_matrix, config,0, config['max_images'] )
    #transforms = get_transforms()

    
    o3d.visualization.draw([pcds[1], pcds[2]])
    #np.save("data/transforms.npy",transforms)
    start = time.time()
    pose_graph = multiway_registration(pcds, 0, len(pcds), config)
    pose_graph = optimize_posegraph(pose_graph)
    #o3d.io.write_pose_graph("data/pose_graph.json", pose_graph)

    print(time.time()-start)
    vbg = integrade(pose_graph, intrinsics)
    mesh = vbg.extract_triangle_mesh()
    

    o3d.visualization.draw(mesh)
    

main()




