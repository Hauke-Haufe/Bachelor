import open3d as o3d
import open3d.t.pipelines.registration as o3r
import json
import os
from config import FRAGMENT_PATH, SCENE_PATH 
import time


class icp:

    @staticmethod
    def _multiscale_registration(source, target, init_transform):
        
        criteria_list = [
            o3r.ICPConvergenceCriteria(0.0001, 0.0001, 20 ),
            o3r.ICPConvergenceCriteria(0.00001, 0.00001, 15),
            o3r.ICPConvergenceCriteria(0.000001, 0.000001, 10)
        ]
        
        max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
        voxle_sizes = o3d.utility.DoubleVector([0.03, 0.02, 0.01])
        init_transform = o3d.core.Tensor.eye(4)
        estimation = o3r.TransformationEstimationPointToPlane()

        result = o3r.multi_scale_icp(source,
                                    target,
                                    voxle_sizes,
                                    criteria_list, 
                                    max_correspondence_distances, 
                                    init_transform, 
                                    estimation)
        
        o3d.visualization.draw([source.transform(result.transformation), target])

        return result.transformation

    @staticmethod
    def _global_registration(source, target, source_fpfh, target_fpfh, config):

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

        o3d.visualization.draw([source, target.transform(result.transformation)])

        return result.transformation


    def run_system(self, parallel= False):

        with open("config.json", "rb") as file:
            config = json.load(file)

        radius_feature = config["voxel_size"] * 6
        num_fragments = len(os.listdir(FRAGMENT_PATH))

        pcds, fpfhs = [], []
        for fragment in os.listdir(FRAGMENT_PATH):
            pcd = o3d.t.io.read_point_cloud(os.path.join(FRAGMENT_PATH, fragment))
            start = time.time()
            fpfh =o3r.compute_fpfh_feature(
                pcd.voxel_down_sample(0.02).cuda() , 
                radius= radius_feature, 
                max_nn = 100)
            
            print(f"fpfh time: {time.time()-start}")
            pcds.append(pcd)
            f = o3d.pipelines.registration.Feature()
            f.data = fpfh.cpu().numpy()
            fpfhs.append(f)


        merged_pcd = pcds[0]
        for i in range(1,num_fragments):

            start = time.time()
            coarse = self._global_registration(pcds[i-1].to_legacy(), 
                                               pcds[i].to_legacy(), 
                                               fpfhs[i-1], 
                                               fpfhs[i],
                                               config)
            
            print(f"Ransac time: {time.time()-start}")

            fine = self._multiscale_registration(pcds[i-1].cuda(),
                                                 pcds[i].cuda(),
                                                 coarse)
            
            merged_pcd.transform(fine)
            merged_pcd += pcds[i]
        
        o3d.visualization.draw(merged_pcd)
        o3d.t.io.write_point_cloud(os.path.join(SCENE_PATH, "scene.pcd"), merged_pcd)


if __name__ == "__main__":

    system = icp()
    system.run_system()




