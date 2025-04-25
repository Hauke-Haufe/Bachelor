import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def show_image(image):

    image = image.as_tensor()
    image = image.numpy()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.imshow(image)

    plt.show()

def plot_pcds(pcds):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    pc = pcds[0]
    for i in range(1,len(pcds)):
        pc += pcds[i]
    pc = pc.point.positions
    ax.scatter(pc.numpy()[:,0],pc.numpy()[:,1], pc.numpy()[:,2])
    plt.show()

def set_up_transformations():

    transforms = np.load("data/transform.npy")

    rel_transforms = []
    for i in range(1, transforms.shape[0]):
        rel_t = transforms[i] @ np.linalg.inv(transforms[i-1])
        rel_transforms.append(rel_t)

    return rel_transforms

def load_from_rsbag(path, which):

    pcds = []
    reader = o3d.t.io.RSBagReader()
    reader.open(path)
    intrinsics = reader.metadata.intrinsics.intrinsic_matrix

    reader.close()
    
    if which:
        reader = o3d.t.io.RSBagReader()
        reader.open(path)
        intrinsics = reader.metadata.intrinsics.intrinsic_matrix
        pcds = []
        frame = reader.next_frame()
        while not reader.is_eof():
            frame = reader.next_frame()
            if not frame.is_empty():
                pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(frame, intrinsics, with_normals = True)
                pcds.append(pcd)

        reader.close()

    else:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(path, repeat_playback=False)
        pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            for i in range(0,100):
                if i% 20== 0:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    depth_image = o3d.t.geometry.Image(depth_image)
                    color_image = o3d.t.geometry.Image(color_image)
                    image = o3d.t.geometry.RGBDImage(color_image, depth_image, aligned = True)

                    #pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth_image, intrinsics, with_normals = True)
                    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(image, intrinsics, with_normals = True)
                   
                    minb = o3d.core.Tensor([-1.5,-1.5, 0])
                    maxb = o3d.core.Tensor([1.5,1.5,1.5])
                    bb = o3d.t.geometry.AxisAlignedBoundingBox(minb, maxb)
                    pcd = pcd.crop(bb)

                    pcd.cuda()
                    pcds.append(pcd)
        finally:
            pass

    return pcds

def set_up_transformations(transforms):

    rel_transforms = []
    for i in range(1, len(transforms)):
        rel_t = transforms[i] @ np.linalg.inv(transforms[i-1])
        rel_t = o3d.core.Tensor(rel_t).cuda()
        rel_transforms.append(rel_t)

    return rel_transforms

def load_from_ply():
    
    ply_files = [f for f in os.listdir("data") if f.endswith(".ply")]

    pcds = []
    for file in ply_files:
        file_path = os.path.join("data", file)
        pcd = o3d.t.io.read_point_cloud(file_path , format = 'ply')
        if not pcd.is_empty():
            pcd.estimate_normals()
            pcd = pcd.cuda()
            pcds.append(pcd)
            
    return pcds

def stream_bag(bag_file):

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)

    pipeline.start(config)
    pc = rs.pointcloud()
    decimation = rs.decimation_filter()
    colorizer = rs.colorizer()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)

            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1,3)
            col = np.asanyarray(color_frame.get_data()).reshape(-1,3) / 255.0

            pcd = o3d.t.geometry.PointCloud()
            pcd.point.positions = o3d.core.Tensor(vtx, dtype = o3d.core.Dtype.Float32)
            pcd.point.colors = o3d.core.Tensor(col, dtype = o3d.core.Dtype.Float32)
            pcd

            vis.clear_geometries()
            vis.add_geometry(pcd.to_legacy())
            vis.poll_events()
            vis.update_renderer()
    except KeyboardInterrupt:
        vis.destroy_window()
        pipeline.stop()
    
def get_pointcloud(color_frame, depth_frame, intrinsics):

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_image = o3d.t.geometry.Image(depth_image)
    color_image = o3d.t.geometry.Image(color_image)
    image = o3d.t.geometry.RGBDImage(color_image, depth_image, aligned = True)

    #pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth_image, intrinsics, with_normals = True)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(image, intrinsics, with_normals = True)
    
    minb = o3d.core.Tensor([-2.0, -2.0, 0])
    maxb = o3d.core.Tensor([2.0,2.0,2.0])
    bb = o3d.t.geometry.AxisAlignedBoundingBox(minb, maxb)
    #pcd = pcd.crop(bb)
    
    return pcd.cuda()

def estimate_transform(gyro_data, acc_data, timestamp, index):

    prev_timestamp = gyro_data[index][0]
    Filter = IMUCalmanFilter(dt = 0.01)
    while gyro_data[index][0] < timestamp and index < acc_data.shape[0]-1:

                    cur_timestamp = gyro_data[index][0]
                    dt = cur_timestamp -prev_timestamp

                    if dt != 0:
                        imu_data = np.concatenate((acc_data[index][1:4], gyro_data[index][1:4])).reshape((6,1))
                        Filter.set_dt(dt)
                        Filter.predict(imu_data)
                        Filter.update(imu_data)
                        
                    index += 1
                    prev_timestamp = cur_timestamp 

    transform = Filter.get_transform()
    return transform , index
    
def load_bag_imu(path, config):

    reader = o3d.t.io.RSBagReader()
    reader.open(path)
    intrinsics = reader.metadata.intrinsics.intrinsic_matrix
    stream_lenght = reader.metadata.stream_length_usec /1000000
    reader.close()

    freq = config["key_frame_freq"]
    acc_data = np.load("data/acceleration.npy")
    gyro_data = np.load("data/gyrodata.npy")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device_from_file(path, repeat_playback=False)

    pipeline.start(rs_config)
    align = rs.align(rs.stream.color)

    pcds = []
    transforms = []

    try:
        index =  0
        i = 0 
        passed_time = 0

        while passed_time < stream_lenght:

            frames = pipeline.wait_for_frames()
            if i% freq== 0:
                frames.keep()

                if not frames.get_depth_frame() or not frames.get_color_frame():
                    continue

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                timestamp = color_frame.timestamp / 1000
                
                transform, index = estimate_transform(gyro_data, acc_data, timestamp, index)
                pcd = get_pointcloud(color_frame, depth_frame, intrinsics)
                
                transforms.append(np.linalg.inv(transform))
                pcds.append(pcd) 

                if i == 0:
                    prev_timestamp = timestamp
                else:
                    passed_time += timestamp -prev_timestamp
                    prev_timestamp = timestamp
                
            i += 1

    finally:
        pass

    pipeline.stop()
    
    return pcds, transforms
