import open3d.visualization as vis
import open3d as o3d
import numpy as np
import os


def plot_graph(pcd, posegraphs, traj_only, without_frames):

    linesets = []
    frames = []
    for posegraph in posegraphs:

        trajectories = [posegraph.nodes[i].pose for i in range(len(posegraph.nodes))]
        camera_centers = [T[:3, 3] for T in trajectories]

        
        for T in trajectories:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            frame.transform(T)
            frames.append(frame)

        traj_lines = []
        traj_colors = []
        for edge in posegraph.edges:
            traj_lines.append([edge.source_node_id, edge.target_node_id])
            
            if edge.uncertain:
                
                traj_colors.append([0,0,1])
                
            else: 
                traj_colors.append([1,0,0])

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(camera_centers)
        lineset.lines = o3d.utility.Vector2iVector(traj_lines)
        lineset.colors = o3d.utility.Vector3dVector(traj_colors)

        linesets.append(lineset)

    if traj_only:
        if without_frames:
            vis.draw_geometries(linesets)
        else:#
            vis.draw_geometries(linesets+frames)
    else:
        if without_frames:
            vis.draw([pcd] + linesets)
        else:
            vis.draw([pcd] + linesets+frames)

    pass

def test_graph(graph):
    prev = np.eye(4)

    for i in range(len(graph.nodes)):
        node = graph.nodes[i]
        curr = node.pose
        dt = prev-curr

        if np.linalg.norm(dt[:3,3]) > 0.2:
            print(i)

        prev = curr

if __name__ == "__main__":
    
    """pcd = o3d.io.read_point_cloud(f"build/pointcloud_masked_odometry.pcd")
    graph = o3d.io.read_pose_graph("build/graph_masked_odometry.json")

    plot_graph(pcd, [graph], False, True)"""

    pcd = o3d.io.read_point_cloud("odometry/cpp/maskout.pcd")
    o3d.visualization.draw([pcd])
