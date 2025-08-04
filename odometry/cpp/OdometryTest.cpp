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

t::pipelines::odometry::OdometryResult DOdometry(Tum_dataset& dataset, 
                                                 int i, int j, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                 bool& sucess,
                                                 core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))) ){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDMImage(j);

    try{
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
        return result;
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        return t::pipelines::odometry::OdometryResult();
    }
}

t::pipelines::odometry::OdometryResult TOdometry(Tum_dataset& dataset, 
                                                 int i, int j,  t::pipelines::odometry::Method method, 
                                                 core::Device device, 
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria,
                                                 bool& sucess, 
                                                 core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDMImage(j);

    try{
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
        return result;
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        return t::pipelines::odometry::OdometryResult();
    }
}

t::pipelines::odometry::OdometryResult SOdometry(Tum_dataset& dataset, 
                                                 int i, int j, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                 bool& sucess,
                                                 core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDImage(j);

    try{
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
        return result; 
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        return t::pipelines::odometry::OdometryResult();
    }

}

t::pipelines::odometry::OdometryResult Odometry(Tum_dataset& dataset, 
                                                int i, int j, t::pipelines::odometry::Method method, 
                                                core::Device device, 
                                                std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                bool& sucess,
                                                core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDImage(j);

    //auto Pcd = t::geometry::PointCloud::CreateFromDepthImage(source.depth_, dataset.get_intrinsics(device), 
    //    core::Tensor::Eye(4, core::Float32, ((open3d::core::Device)("CPU:0"))), 1000.0f, 10.0f);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((Pcd).ToLegacy())});
    try{
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
    return result; 
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
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

    for(int i = 0; i< dataset.get_size() -1 ; i++){

        t::pipelines::odometry::OdometryResult result;
        
        switch (m_mehtod)
        {
            case MaskMethod::SourceMask:
                result = SOdometry(dataset, i, i+1, method, device, critirias, sucess);
                break;

            case MaskMethod::TargetMask:
                result = TOdometry(dataset, i, i+1, method, device, critirias, sucess);
                break;
            
            case MaskMethod::CompleteMask:
                result = DOdometry(dataset, i, i+1,  method, device, critirias, sucess);
                break;

            case MaskMethod::NoMask:
                result = Odometry(dataset, i, i+1, method, device, critirias, sucess);
                break;
        }

        t::geometry::RGBDImage source = dataset.get_RGBDImage(i);
        t::geometry::RGBDImage target = dataset.get_RGBDImage(i+1);

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

    std::cout << "ATE: " << dataset.ComputeATE(Trajectory, false)<< std::endl;
    std::cout << "RPE: " << dataset.ComputeRPE(Trajectory, 1)<< std::endl;

    graph.Save("graph.json");
    //t::io::WritePointCloud("pointcloud.pcd", vgb->ExtractPointCloud());
};

void ComputeMetricsOptimize(SubDataset data,t::pipelines::odometry::Method method, MaskMethod m_mehtod, core::Device  device){

    auto dataset = Tum_dataset(data);
    auto Trajectory = std::vector<core::Tensor>();

    auto dtype = core::Dtype::Float32;

    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {6,3,1};
    core::Tensor pose = dataset.get_init_pose();
    

    Trajectory.push_back(pose);
    Posegraph<Open3dPosegraphBackend> graph(pose);
    bool sucess, uncertain;

    for(int i = 0; i< dataset.get_size() -1 ; i++){
        for (int j = i+1; j< dataset.get_size()  ; j = j + 200){

            t::pipelines::odometry::OdometryResult result;
            
            switch (m_mehtod){
            case MaskMethod::SourceMask:
                result = SOdometry(dataset, i, i+1, method, device, critirias, sucess);
                break;

            case MaskMethod::TargetMask:
                result = TOdometry(dataset, i, i+1, method, device, critirias, sucess);
                break;
            
            case MaskMethod::CompleteMask:
                result = DOdometry(dataset, i, i+1,  method, device, critirias, sucess);
                break;

            case MaskMethod::NoMask:
                result = Odometry(dataset, i, i+1, method, device, critirias, sucess);
                break;
            }

            t::geometry::RGBDImage source = dataset.get_RGBDImage(i);
            t::geometry::RGBDImage target = dataset.get_RGBDImage(j);
            
            if (i+1==j){
                auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(
                    source.depth_.To(device), target.depth_.To(device),
                    dataset.get_intrinsics(core::Device("CPU:0")), result.transformation_, 0.1, 5000.0f, 5.0f);
                
                graph.AddOdometryEdge(result.transformation_, information, i, j, false);
                graph.AddNode(pose.Inverse());
                Trajectory.push_back(pose);
                pose = pose.Matmul(result.transformation_);
            }
            else if (sucess && j > i+1){
               auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(
                    source.depth_.To(device), target.depth_.To(device),
                    dataset.get_intrinsics(core::Device("CPU:0")), result.transformation_, 0.1, 5000.0f, 5.0f);
                
                graph.AddOdometryEdge(result.transformation_, information, i, j, true); 
            }
        }    
    } 
    std::cout << "Odometry finished" << std::endl;
    graph.Optimize();
    std::cout << "Optimization finished" <<std::endl;

    auto o3d_posegraph = graph.GetPoseGraph();
    auto intrinsics = dataset.get_intrinsics(core::Device("CPU:0"));
    //auto vgb = integrate(o3d_posegraph, dataset.get_colorfiles_paths(), dataset.get_depthfiles_paths(), intrinsics, 5000.0, 5.0);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((vgb->ExtractPointCloud()).ToLegacy())});

    //graph.Save("graph.json");
    //t::io::WritePointCloud("pointcloud.pcd", vgb->ExtractPointCloud());
    std::cout << "ATE: " << dataset.ComputeATE(Trajectory, false)<< std::endl;
    std::cout << "RPE: " << dataset.ComputeRPE(Trajectory, 1)<< std::endl;
};

int main() {
    
    std::ifstream config_file("configs/odometry_config.json");
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
 