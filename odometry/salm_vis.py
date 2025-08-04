from slam_visualizer import plot_graph
import open3d as o3d

def plot_static():
    trajec_only = False
    without_frames = True

    pcd = o3d.io.read_point_cloud("presentation_data/recon_data/walking_static_raw.pcd")
    graph = o3d.io.read_pose_graph("presentation_data/recon_data/walking_static_raw.json")

    plot_graph(pcd, [graph], False, True)

    pcd = o3d.io.read_point_cloud("presentation_data/recon_data/walking_static_maskout.pcd")
    graph = o3d.io.read_pose_graph("presentation_data/recon_data/walking_static_maskout.json")

    plot_graph(pcd, [graph], trajec_only, without_frames)

def plot_xyz():

    trajec_only = False
    without_frames = True

    pcd = o3d.io.read_point_cloud("presentation_data/recon_data/walking_xyz_raw.pcd")
    graph = o3d.io.read_pose_graph("presentation_data/recon_data/walking_xyz_raw.json")

    plot_graph(pcd, [graph], False, True)


    pcd = o3d.io.read_point_cloud("presentation_data/recon_data/walking_xyz_maskout.pcd")
    graph = o3d.io.read_pose_graph("presentation_data/recon_data/walking_xyz_maskout.json")

    plot_graph(pcd, [graph], trajec_only, without_frames)

plot_xyz()