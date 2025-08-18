#include "CONFIG.h"
#include "TestDataset.h"
#include "pose_graph.h"

#include <open3d/core/Tensor.h>
#include <open3d/t/pipelines/odometry/RGBDMOdometry.h>
#include <open3d/t/pipelines/odometry/RGBDOdometry.h>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include<chrono>

using namespace open3d;

std::unique_ptr<t::geometry::VoxelBlockGrid> create_vgb(){

    std::vector<std::string> atr_names = {"tsdf", "weight", "color"};
    std::vector<core::Dtype> atr_types = {core::Dtype::Float32,core::Dtype::UInt16, core::Dtype::UInt16};
    std::vector<core::SizeVector> channels = {{1}, {1}, {3}};
    core::Device device("CPU:0");

    return std::make_unique<t::geometry::VoxelBlockGrid>(
        atr_names, 
        atr_types,
        channels,
        0.01,
        16,
        1000,
        device
    );

}

std::unique_ptr<t::geometry::VoxelBlockGrid> integrate(
    pipelines::registration::PoseGraph& Posegraph,
    const std::vector<fs::path>& color_images,
    const std::vector<fs::path>& depth_images, 
    core::Tensor& instrinsics, 
    float depth_scale,
    float depth_max
    ){
    
    std::unique_ptr<t::geometry::VoxelBlockGrid> vgb;
    vgb = create_vgb();

    t::geometry::Image color_image;
    t::geometry::Image depth_image;
    core::Device cuda_device(core::Device::DeviceType::CPU, 0);
    
    for(int i = 0; i < color_images.size()-1; i++){

        t::io::ReadImage(color_images[i], color_image);
        t::io::ReadImage(depth_images[i], depth_image);
        
        auto pose = Posegraph.nodes_[i].pose_;
        Eigen::Matrix4d inverse = pose.inverse();

        auto inverse_t = core::eigen_converter::EigenMatrixToTensor(inverse);

        auto frustum_block_coords = vgb->GetUniqueBlockCoordinates(
            depth_image.To(cuda_device),
            instrinsics,
            inverse_t, 
            depth_scale,
            depth_max
        );

        vgb->Integrate(
            frustum_block_coords,
            depth_image.To(cuda_device),
            color_image.To(cuda_device),
            instrinsics,
            inverse_t,
            depth_scale,
            depth_max
        );

    }

    return vgb;
}

t::pipelines::odometry::OdometryResult DOdometry(Tum_dataset& dataset, 
                                                 int i, int j, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                 bool& sucess,
                                                 double& time,
                                                 core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDMImage(j);

    try{
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(),
                init, 
                5000.0F,
                7.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        sucess = true;
        auto end =  std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        time = duration.count();
        return result;
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        time = 0;
        return t::pipelines::odometry::OdometryResult();
    }
}

t::pipelines::odometry::OdometryResult TOdometry(Tum_dataset& dataset, 
                                                 int i, int j, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                 bool& sucess,
                                                 double& time,
                                                 core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDMImage(j);

    try{
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device), 
                dataset.get_intrinsics(), 
                init, 
                5000.0F,
                7.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        sucess = true;
        auto end =  std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        time = duration.count();
        return result;
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        time = 0;
        return t::pipelines::odometry::OdometryResult();
    }
}

t::pipelines::odometry::OdometryResult SOdometry(Tum_dataset& dataset, 
                                                 int i, int j, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                 bool& sucess,
                                                 double& time,
                                                 core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDImage(j);

    try{
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(),
                init, 
                5000.0F,
                7.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        sucess = true;
        auto end =  std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        time = duration.count();
        return result; 
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        time = 0;
        return t::pipelines::odometry::OdometryResult();
    }

}

t::pipelines::odometry::OdometryResult Odometry(Tum_dataset& dataset, 
                                                int i, int j, t::pipelines::odometry::Method method, 
                                                core::Device device, 
                                                std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                bool& sucess,
                                                double& time,
                                                core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDImage(j);

    //auto Pcd = t::geometry::PointCloud::CreateFromDepthImage(source.depth_, dataset.get_intrinsics(device), 
    //    core::Tensor::Eye(4, core::Float32, ((open3d::core::Device)("CPU:0"))), 1000.0f, 10.0f);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((Pcd).ToLegacy())});
    try{
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t::pipelines::odometry::RGBDOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(),
                init, 
                5000.0F,
                7.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        sucess = true;
        auto end =  std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        time = duration.count();
    return result; 
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        time = 0;
        return t::pipelines::odometry::OdometryResult();
    }
}

void ComputeMetrics(SubDataset data, t::pipelines::odometry::Method method, MaskMethod m_mehtod, core::Device  device){

    auto dataset = Tum_dataset(data);
    auto Trajectory = std::vector<core::Tensor>();
    Trajectory.resize(dataset.get_size());

    auto dtype = core::Dtype::Float32;

    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {10,10,10};
    core::Tensor pose = dataset.get_init_pose();
    
    Trajectory[0]= pose;
    Posegraph<Open3dPosegraphBackend> graph(pose);
    bool sucess;

    double time = 0;
    int count = 0;

    for(int i = 0; i< dataset.get_size() -1 ; i++){

        t::pipelines::odometry::OdometryResult result;
        
        double meas_time;
        switch (m_mehtod)
        {
            case MaskMethod::SourceMask:
                result = SOdometry(dataset, i, i+1, method, device, critirias, sucess, meas_time);
                break;

            case MaskMethod::TargetMask:
                result = TOdometry(dataset, i, i+1, method, device, critirias, sucess, meas_time);
                break;
            
            case MaskMethod::CompleteMask:
                result = DOdometry(dataset, i, i+1,  method, device, critirias, sucess, meas_time);
                break;

            case MaskMethod::NoMask:
                result = Odometry(dataset, i, i+1, method, device, critirias, sucess, meas_time);
                break;
        }
        time += meas_time;

        t::geometry::RGBDImage source = dataset.get_RGBDImage(i);
        t::geometry::RGBDImage target = dataset.get_RGBDImage(i+1);

        if (sucess){
            count++;
        }
        //information muss auch noch geändert werden
        auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(
            source.depth_.To(device), target.depth_.To(device),
            dataset.get_intrinsics(core::Device("CPU:0")), result.transformation_, 0.1, 5000.0f, 5.0f);
        
        //std::cout << dataset.get_timestamp(i) << std::endl;

        pose = pose.Matmul(result.transformation_);
        graph.AddOdometryEdge(result.transformation_, information, i, i+1, false);
        graph.AddNode(pose.Inverse());
        
        Trajectory[i +1] = pose;
    } 

    auto o3d_posegraph = graph.GetPoseGraph();
    auto intrinsics = dataset.get_intrinsics(core::Device("CPU:0"));
    //auto vgb = integrate(o3d_posegraph, dataset.get_colorfiles_paths(), dataset.get_depthfiles_paths(), intrinsics, 5000.0, 5.0);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((vgb->ExtractPointCloud()).ToLegacy())});

    nlohmann::json returnJson;
    returnJson["ATE_Trans"] = dataset.ComputeATETrans(Trajectory, false);
    returnJson["ATE_Rot"] = dataset.ComputeATERot(Trajectory);
    returnJson["RPE_Trans"] = dataset.ComputeRPETrans(Trajectory, 1);
    returnJson["RPE_Rot"] = dataset.ComputeRPERot(Trajectory, 1);
    returnJson["total_time"] = time ;
    returnJson["avg_time"] = time/count;

    std::ofstream file("data/output.json");
    file << returnJson.dump(4); 
    file.close();

    //graph.Save("graph.json");
    //t::io::WritePointCloud("pointcloud.pcd", vgb->ExtractPointCloud());
};

int main() {
    
    std::ifstream config_file("data/testConfigs/odometry_config.json");
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config.json");
    }

    nlohmann::json config;
    config_file >> config;

    core::Device device = get_device(config["device"]);
    t::pipelines::odometry::Method method = get_method(config["method"]);
    MaskMethod mask_mehtod = get_maskmethod(config["maskingmethod"]);
    SubDataset dataset_type = get_dataset(config["dataset"]);

    std::cout << fmt::format("On config: Device {}, Method {}, Maskoutmethod {}, Dataset {}",
        config["device"].get<std::string>(),  config["method"].get<std::string>(), 
        config["maskingmethod"].get<std::string>(),config["dataset"].get<std::string>())<< std::endl;

    ComputeMetrics(dataset_type, method, mask_mehtod, device);

}
 