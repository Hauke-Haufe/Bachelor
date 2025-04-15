import pyzed.sl as sl
import numpy as np
import os
import cv2
from datetime import datetime


def camera_initHD():
    # Create a ZED camera object
    zed = sl.Camera()

    # camera parameters
    
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY  
    
    #tracking prameters
    
    tracking_parameters = sl.PositionalTrackingParameters()
    tracking_parameters.enable_imu_fusion = True
    
    #spatial mapping paramters
    
    mapping_parameters = sl.SpatialMappingParameters()
    mapping_parameters.resolution_meter = mapping_parameters.get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
    mapping_parameters.range_meter = mapping_parameters.get_range_preset(sl.MAPPING_RANGE.SHORT)
    mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    mapping_parameters.save_texture = False

    
    err = zed.open(init_params)
    if (err != sl.ERROR_CODE.SUCCESS) :
        zed.close()
        exit(-1)
        
    err = zed.enable_positional_tracking(tracking_parameters)
    if (err != sl.ERROR_CODE.SUCCESS):
        zed.close()
        exit(-1)
        
    ''' err = zed.enable_spatial_mapping(mapping_parameters)
    if (err != sl.ERROR_CODE.SUCCESS):
        zed.close()
        exit(-1)'''
    
        
    return zed

def camera_init():
    # Create a ZED camera object
    zed = sl.Camera()

    # camera parameters
    
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY  
    
    #tracking prameters
    
    tracking_parameters = sl.PositionalTrackingParameters()
    tracking_parameters.enable_imu_fusion = True
    
    #spatial mapping paramters
    
    mapping_parameters = sl.SpatialMappingParameters()
    mapping_parameters.resolution_meter = mapping_parameters.get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
    mapping_parameters.range_meter = mapping_parameters.get_range_preset(sl.MAPPING_RANGE.SHORT)
    mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    mapping_parameters.save_texture = False

    
    err = zed.open(init_params)
    if (err != sl.ERROR_CODE.SUCCESS) :
        zed.close()
        exit(-1)
        
    err = zed.enable_positional_tracking(tracking_parameters)
    if (err != sl.ERROR_CODE.SUCCESS):
        zed.close()
        exit(-1)
        
    ''' err = zed.enable_spatial_mapping(mapping_parameters)
    if (err != sl.ERROR_CODE.SUCCESS):
        zed.close()
        exit(-1)'''
    
        
    return zed

def capture_pointcloud():
    zed = camera_init()
    i = 1 
    count = 2

    runtime_parameters = sl.RuntimeParameters()
    sensor = sl.SensorsData()
    imu_data = sl.IMUData()
    transform = sl.Transform()
    pose = np.zeros((count,4,4))

    mapping_parameters = sl.SpatialMappingParameters()
    mapping_parameters.resolution_meter = mapping_parameters.get_resolution_preset(sl.MAPPING_RESOLUTION.LOW)
    mapping_parameters.range_meter = mapping_parameters.get_range_preset(sl.MAPPING_RANGE.SHORT)
    mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    mapping_parameters.save_texture = False
 
    for j in range(0,count):
        err = zed.enable_spatial_mapping(mapping_parameters)
        if (err != sl.ERROR_CODE.SUCCESS):
            zed.close()
            exit(-1)

        while i < 1000:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
                i += 1
        
        zed.get_sensors_data(sensor, sl.TIME_REFERENCE.CURRENT)
        imu_data = sensor.get_imu_data()
        imu_data.get_pose(transform)
        pose[j] = transform.m

        cloud = sl.FusedPointCloud()
        zed.extract_whole_spatial_map(cloud)
        cloud.save(f"data/pointcloud{j}.ply", typeMesh = sl.MESH_FILE_FORMAT.PLY)
        zed.disable_spatial_mapping()

    np.save("data/transform.npy", pose)

def capture_RGBDHD():
    
    path = "/home/nb-messen-07/Desktop/SpatialMapping/capturing/data/zed_data/HD/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path + f"{timestamp}"
    
    os.makedirs( path + "/depth", exist_ok=True)
    os.makedirs( path + "/color", exist_ok=True)

    zed = camera_init()
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    '''sensor = sl.SensorsData()
    imu_data = sl.IMUData()
    transform = sl.Transform()
    pose = np.zeros((count,4,4))'''

 
    
    i = 0
    while True:

        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            rgb = image.get_data()
            depth_data = depth.get_data()
            depth_data = np.clip(depth_data,0, 65535).astype(np.uint16) 


            cv2.imwrite(path + f"/depth/image{i}.png", depth_data)
            cv2.imwrite(path +f"/color/image{i}.png", rgb[:,:,:3])
    
        '''zed.get_sensors_data(sensor, sl.TIME_REFERENCE.CURRENT)
        imu_data = sensor.get_imu_data()
        imu_data.get_pose(transform)
        pose[j] = transform.m'''
        i += 1

def capture_RGBDVGA():

    path = "/home/nb-messen-07/Desktop/SpatialMapping/capturing/data/zed_data/VGA/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path + f"{timestamp}"

    os.makedirs( path + "/depth", exist_ok=True)
    os.makedirs( path + "/color", exist_ok=True)

    zed = camera_init()
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    '''sensor = sl.SensorsData()
    imu_data = sl.IMUData()
    transform = sl.Transform()
    pose = np.zeros((count,4,4))'''

 
    
    i = 0
    while True:

        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            rgb = image.get_data()
            depth_data = depth.get_data()
            depth_data = np.clip(depth_data,0, 65535).astype(np.uint16) 


            cv2.imwrite(path + f"/depth/image{i}.png", depth_data)
            cv2.imwrite(path +f"/color/image{i}.png", rgb[:,:,:3])
    
        '''zed.get_sensors_data(sensor, sl.TIME_REFERENCE.CURRENT)
        imu_data = sensor.get_imu_data()
        imu_data.get_pose(transform)
        pose[j] = transform.m'''
        i += 1



#capture_RGBDVGA()
capture_RGBDHD()

