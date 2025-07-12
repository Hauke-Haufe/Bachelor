import open3d as o3d
import open3d.t.pipelines.registration as o3r
from posegraph import Posegraph
import json
import os
from config import *
import numpy as np

class icp:

    @staticmethod
    def multiscale_registration(source, target, init_transform, colored):
        
        criteria_list = [
            o3r.ICPConvergenceCriteria(0.0001, 0.0001, 100 ),
            o3r.ICPConvergenceCriteria(0.00001, 0.00001, 100),
            o3r.ICPConvergenceCriteria(0.000001, 0.000001, 100)
        ]
        
        max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
        voxle_sizes = o3d.utility.DoubleVector([0.03, 0.02, 0.01])
        scale_param = 0.1

        if colored:
            estimation = o3r.TransformationEstimationForColoredICP(
                o3r.robust_kernel.RobustKernel(
                    o3r.robust_kernel.RobustKernelMethod.TukeyLoss, 
                    scale_param
                )
            )
        else:
            estimation = o3r.TransformationEstimationPointToPlane(
                o3r.robust_kernel.RobustKernel(
                    o3r.robust_kernel.RobustKernelMethod.TukeyLoss, 
                    scale_param
                )
            )


        result = o3r.multi_scale_icp(source,
                                    target,
                                    voxle_sizes,
                                    criteria_list, 
                                    max_correspondence_distances, 
                                    init_transform, 
                                    estimation)
        
        #o3d.visualization.draw([source.transform(result.transformation), target])

        return result.transformation

    @staticmethod
    def global_registration(source, target, source_fpfh, target_fpfh, config):

        distance_threshold = config["voxel_size"] * 1.5
        source = source.voxel_down_sample(voxel_size = 0.02)
        target = target.voxel_down_sample(voxel_size = 0.02)

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.9999)
        )

        #o3d.visualization.draw([source, target.transform(result.transformation)])

        return result.transformation

    def run_system(self, parallel= False):

        with open("config.json", "rb") as file:
            config = json.load(file)


        fragments = [f for f in os.listdir(FRAGMENT_PATH)  if f.endswith(".pcd")]
        fragments = sorted(fragments, reverse=False)
        graphs = [f for f in os.listdir(FRAGMENT_PATH)  if not f.endswith(".pcd")]
        graphs = sorted(graphs, reverse=False)

        pcds, inits = [], []
        for fragment in fragments:
            pcd = o3d.t.io.read_point_cloud(os.path.join(FRAGMENT_PATH, fragment))
            colors = pcd.point.colors
            colors = colors.to(o3d.core.Dtype.Float32) / 255.0
            pcd.point.colors = colors
            pcds.append(pcd)

        overlab_ids = [0]
        overlab_id = 0
        for g in graphs:
            graph = Posegraph(config["posegraph_backend"], np.eye(4), config["imu"])
            graph.load(os.path.join(FRAGMENT_PATH, g))
            old_overlap_id = overlab_id
            overlab_id += config["frames_per_frag"] - config["frag_overlap"]
            inits.append(graph[overlab_id-old_overlap_id])
            overlab_ids.append(overlab_id)


        transforms = []
        for i in range(1,len(fragments)):

            fine = self.multiscale_registration(
                pcds[i-1].cuda(),pcds[i].cuda(), np.linalg.inv(inits[i-1]),
                config["colored_icp"])
            
            o3d.visualization.draw([pcds[i-1].clone().transform(fine.numpy()), pcds[i]])
            transforms.append(fine)
        
        if config["posegraph_backend"] == "gtsam":

            



if __name__ == "__main__":

    system = icp()
    system.run_system()




