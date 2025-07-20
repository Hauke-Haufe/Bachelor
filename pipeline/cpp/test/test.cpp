#include "test.h"
#include "test_dataset.h"
#include "../pose_graph.h"
#include "../Integration.h"


#include <fmt/core.h>
#include <open3d/core/TensorKey.h>
#include <open3d/t/pipelines/odometry/RGBDOdometry.h>
#include <open3d/core/Device.h>
#include <open3d/core/Dtype.h>
#include <open3d/t/io/ImageIO.h>
#include <open3d/pipelines/registration/GlobalOptimization.h>
#include "open3d/t/pipelines/odometry/RGBDMOdometry.h"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <chrono>

using namespace open3d;
namespace fs = std::filesystem;

int CountFilesDir(fs::path dirPath){

    std::size_t count = 0;

    for(auto& entry: fs::directory_iterator(dirPath)){
        ++count;
    }

    return count;
}

bool IsLargeRotation(const open3d::core::Tensor& trans,
                     double angle_threshold_deg = 10.0) {
    using namespace open3d;

    // Convert to Eigen
    Eigen::Matrix4d rel = core::eigen_converter::TensorToEigenMatrixXd(
                         trans.To(core::Device("CPU:0")));



    // Extract rotation matrix
    Eigen::Matrix3d R = rel.block<3, 3>(0, 0);

    // Compute rotation angle from trace(R) = 1 + 2cos(theta)
    double trace_R = R.trace();
    double cos_theta = std::clamp((trace_R - 1.0) / 2.0, -1.0, 1.0);  // Numerical safety
    double angle_rad = std::acos(cos_theta);
    double angle_deg = angle_rad * 180.0 / M_PI;

    return angle_deg > angle_threshold_deg;
}


bool IsLargeTranslation(const open3d::core::Tensor& transformation,
                        float threshold = 0.3f) {  // adjust threshold as needed
    // Assume shape [4, 4] and Float32 or Float64
    if (transformation.GetShape() != open3d::core::SizeVector{4, 4}) {
        throw std::runtime_error("Transformation must be a 4x4 matrix.");
    }

    // Copy to CPU for access
    auto T_cpu = transformation.To(open3d::core::Device("CPU:0"));

    // Extract translation components (last column)
    float tx = T_cpu[0][3].Item<double>();
    float ty = T_cpu[1][3].Item<double>();
    float tz = T_cpu[2][3].Item<double>();

    float translation_norm = std::sqrt(tx * tx + ty * ty + tz * tz);

    if  (translation_norm > threshold){
        float t = 1;
    }

    return translation_norm > threshold;
}


t::pipelines::odometry::OdometryResult DOdometry(Tum_dataset& dataset, 
                                                 int i, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDMImage(i+1);

    try{
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(device),
                core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
                5000.0F,
                3.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        return result;
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        return t::pipelines::odometry::OdometryResult();
    }
}

t::pipelines::odometry::OdometryResult TOdometry(Tum_dataset& dataset, 
                                                 int i, t::pipelines::odometry::Method method, 
                                                 core::Device device, 
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDMImage(i+1);

    try{
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device), 
                dataset.get_intrinsics(device), 
                core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
                5000.0F,
                3.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        return result;
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        return t::pipelines::odometry::OdometryResult();
    }
}

t::pipelines::odometry::OdometryResult SOdometry(Tum_dataset& dataset, 
                                                 int i, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDImage(i+1);

    try{
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(device),
                core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
                5000.0F,
                3.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        return result; 
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        return t::pipelines::odometry::OdometryResult();
    }

}

t::pipelines::odometry::OdometryResult Odometry(Tum_dataset& dataset, 
                                                int i, t::pipelines::odometry::Method method, 
                                                core::Device device, 
                                                std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDImage(i+1);

    //auto Pcd = t::geometry::PointCloud::CreateFromDepthImage(source.depth_, dataset.get_intrinsics(device), 
    //    core::Tensor::Eye(4, core::Float32, ((open3d::core::Device)("CPU:0"))), 1000.0f, 10.0f);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((Pcd).ToLegacy())});
    try{
    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(source.To(device), 
            target.To(device),
            dataset.get_intrinsics(device),
            core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
            5000.0F,
            3.0F,
            critiria,
            method,
            open3d::t::pipelines::odometry::OdometryLossParams());

    return result; 
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        return t::pipelines::odometry::OdometryResult();
    }
}

void test_masked_odometry(SubDataset data,
                          t::pipelines::odometry::Method method, 
                          t::pipelines::odometry::MaskMethod m_mehtod,
                          core::Device  device){

    auto dataset = Tum_dataset(data);
    auto Trajectory = std::vector<core::Tensor>();

    auto dtype = core::Dtype::Float32;
    const auto init_trans = core::Tensor::Eye(4,core::Dtype::Float64 ,core::Device("CPU:0"));

    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {6,3,1};
    core::Tensor pose = dataset.get_init_pose();
    
    Trajectory.push_back(pose);
    Posegraph<Open3dPosegraphBackend> graph(pose);
    
    for(int i = 0; i< dataset.get_size() -1 ; i++){

        t::pipelines::odometry::OdometryResult result;
        
        switch (m_mehtod)
        {
        case t::pipelines::odometry::MaskMethod::SourceMask:
            result = SOdometry(dataset, i, method, device, critirias);
            break;

        case t::pipelines::odometry::MaskMethod::TargetMask:
            result = TOdometry(dataset, i, method, device, critirias);
            break;
        
        case t::pipelines::odometry::MaskMethod::CompleteMask:
            result = DOdometry(dataset, i, method, device, critirias);
            break;
        }
        result= Odometry(dataset, i, method, device, critirias);
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
        
        Trajectory.push_back(pose);
    } 

    graph.Save("data/test/graph.json");

    pipelines::registration::PoseGraph o3d_graph;
    io::ReadPoseGraph("data/test/graph.json", o3d_graph);

    auto intrinsics = dataset.get_intrinsics(core::Device("CPU:0"));
    auto vgb = integrate(o3d_graph, dataset.get_colorfiles_paths(), dataset.get_depthfiles_paths(), intrinsics, 5000.0, 5.0);
    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((vgb->ExtractPointCloud()).ToLegacy())});

    std::cout << "realative ATE: " << dataset.ComputeATE(Trajectory, true)<< std::endl;
    std::cout << "RPE: " << dataset.ComputeRPE(Trajectory, 1)<< std::endl;
};

void test_slam(SubDataset data, core::Device device, SlamMethod method){

    float depthscale = 5000.0;
    float depth_max = 7.0;
    
    auto dataset = Tum_dataset(data);
    auto Trajectory = std::vector<core::Tensor>();

    t::pipelines::slam::Model model(0.01, 16, 1000, dataset.get_init_pose(), device, true);
    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {10,10,10};
    core::Tensor pose = dataset.get_init_pose();
    Trajectory.push_back(pose);
    t::pipelines::slam::Frame ray_cast_frame(480, 640, dataset.get_intrinsics(), device);
    t::pipelines::slam::Frame source_frame(480, 640, dataset.get_intrinsics(), device);
    t::geometry::RGBDMImage source =  dataset.get_RGBDMImage(0).To(device);
    source_frame.SetDataFromImage("color", source.color_);
    source_frame.SetDataFromImage("depth", source.depth_);
    source_frame.SetDataFromImage("mask", source.mask_);
    model.UpdateFramePose(0, dataset.get_init_pose());

    for (int i = 0; i < 10; i++){
        switch (method)
        {
        case SlamMethod::WeightMask:
            model.WeightMaskedIntegrate(source_frame, depthscale, depth_max);
            break;
        
        case SlamMethod::Raw:
            model.Integrate(source_frame, depthscale, depth_max);
            break;
        
        case SlamMethod::Masked:
            model.MaskedIntegrate(source_frame, depthscale, depth_max);
            break;
        }
    }
    
    Posegraph<Open3dPosegraphBackend> graph(pose);

    Trajectory.push_back(pose);
    t::pipelines::odometry::OdometryResult result;
    core::Tensor current_T_frame_to_world;
    float weight_threshold;

    for(int i = 1; i< dataset.get_size() -1 ; i++){
        
        weight_threshold = std::min(i*1.0f, 15.0f);
        t::geometry::RGBDMImage source = dataset.get_RGBDMImage(i).To(device);

        source_frame.SetDataFromImage("color", source.color_);
        source_frame.SetDataFromImage("depth", source.depth_);

        source_frame.SetDataFromImage("mask", source.mask_);
        model.SynthesizeModelFrame(ray_cast_frame, depthscale, 0.1, depth_max, 8.0f, true, weight_threshold); 

        current_T_frame_to_world = model.GetCurrentFramePose(); 
        bool succes;
        try{ 
            switch (method){
                case SlamMethod::WeightMask:
                    result = model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;
                case SlamMethod::Raw:
                    result = model.TrackFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;
                case SlamMethod::Masked:
                    result =  model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;}
                succes = true;
            
            auto t = t::pipelines::odometry::ComputeResidualMap(dataset.get_RGBDImage(i-1).To(device),
                    dataset.get_RGBDImage(i).To(device), result.transformation_, dataset.get_intrinsics(),
                t::pipelines::odometry::Method::Hybrid, depthscale, depth_max);
            
            t::io::WriteNpy("residual.npy", t.AsTensor());
        }   
        catch(std::runtime_error){
            std::cout<<"tracking lost at" << dataset.get_timestamp(i) << std::endl;
            result = t::pipelines::odometry::OdometryResult();
            succes = false;
        }

        if (succes){
            model.UpdateFramePose(i, current_T_frame_to_world.Matmul(result.transformation_));
             switch (method){
                case SlamMethod::WeightMask:
                    model.WeightMaskedIntegrate(source_frame, depthscale, depth_max, 8.0, 21, 15);
                    break;
                case SlamMethod::Raw:
                    model.Integrate(source_frame, depthscale, depth_max);
                    break;
                case SlamMethod::Masked:
                    model.MaskedIntegrate(source_frame, depthscale, depth_max);
                    break;}
        }
        else{
            model.UpdateFramePose(i, current_T_frame_to_world);
        }

        if (IsLargeTranslation(result.transformation_, 0.1)){
            std::cout<<i<<std::endl;
        }

        //std::cout << dataset.get_timestamp(i) << std::endl;
        auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(
            source.depth_.To(device), ray_cast_frame.GetDataAsImage("depth"), 
            dataset.get_intrinsics(core::Device("CPU:0")), result.transformation_, 0.1, depthscale, depth_max);
        
        //std::cout << dataset.get_timestamp(i) << std::endl;

        graph.AddOdometryEdge(result.transformation_, information, i, i+1, false);
        graph.AddNode(pose);

        pose = model.GetCurrentFramePose();
        Trajectory.push_back(pose);

    } 

    io::WritePoseGraph("graph.json", graph.GetPoseGraph());

    auto pcd = model.ExtractPointCloud(7.0f);
    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((pcd).ToLegacy())});
    t::io::WritePointCloud("p.pcd", pcd);

    std::cout << "realative ATE: " << dataset.ComputeATE(Trajectory, false)<< std::endl;
    std::cout << "RPE: " << dataset.ComputeRPE(Trajectory, 1)<< std::endl;
}
