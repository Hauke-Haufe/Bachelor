from registration import load_point_cloud
import numpy as np
import open3d as o3d

def remove_ground(pcd):

    ground_plane, ground_points = pcd.segment_plane(distance_threshold = 0.1)
    ground_plane = ground_plane.cpu().numpy()
    a, b, c, d = ground_plane[0], ground_plane[1], ground_plane[2], ground_plane[3]

    if not np.isclose(d,0):
        normal = np.array([a/d, b/d, c/d])
    else :
        normal = np.array([a, b, c])

    ground = pcd.select_by_index(ground_points)
    pcd = pcd.select_by_index(ground_points, invert = True)
    mean = np.mean(ground.point.positions.cpu().numpy())

    return pcd, normal, mean

def triangulate_point_cloud(pcd):
    radii = [0.05,0.2, 0.3, 0.5, 1]

    #glatte variante
    """
    mesh, desity =  o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd.to_legacy(), 
        depth = 8)"""
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd.to_legacy(), 
        o3d.utility.DoubleVector(radii))
    
    return mesh

#testen
def center_poincloud(pcd, normal, point):

    normal = normal / np.linalg.norm(normal)

    axis = np.cross(normal, np.array([0,0,1]))
    angle = np.arccos(np.dot(normal, np.array([0,0,1])))

    if np.isclose(np.linalg.norm(axis),0):
        R = np.eye(4)
    else:
        axis = axis /np.linalg.norm(axis)

        a = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
        
        b= axis @ axis.T
        
        R_mat = np.cos(angle)*np.eye(3) + np.sin(angle)*a +(1-np.cos(angle))*b
        R = np.eye(4)
        R[:3,:3] = R_mat
    
    T = np.eye(4)
    T[:3, 3] = -point

    transform = np.dot(T,R)
    pcd.transform(transform)

    return pcd

def seperate_clusters(pcd):
    labels = pcd.cluster_dbscan(0.1, 3).cpu().numpy() #eps gleich voxel downsampler setzen
    print(np.unique(labels, return_counts = True))

def caluculate_volume():

    pcd = load_point_cloud()
    pcd, normal, mean= remove_ground(pcd[0])
    pcd = center_poincloud(pcd, normal, mean)

    #TODO

    mesh = triangulate_point_cloud(pcd)
    o3d.visualization.draw(mesh)

caluculate_volume()