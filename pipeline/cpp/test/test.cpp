#include "test.h"
#include "test_dataset.h"
#include "test_util.h"
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


t::pipelines::odometry::OdometryResult DOdometry(Tum_dataset& dataset, 
                                                 int i, int j, t::pipelines::odometry::Method method, 
                                                 core::Device device,
                                                 std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                                                 bool& sucess ){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDMImage(j);

    try{
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(),
                core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
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
                                                 bool& sucess){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDMImage(j);

    try{
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device), 
                dataset.get_intrinsics(), 
                core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
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
                                                bool& sucess){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDImage(j);

    try{
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(),
                core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
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
                                                bool& sucess){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDImage(j);

    //auto Pcd = t::geometry::PointCloud::CreateFromDepthImage(source.depth_, dataset.get_intrinsics(device), 
    //    core::Tensor::Eye(4, core::Float32, ((open3d::core::Device)("CPU:0"))), 1000.0f, 10.0f);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((Pcd).ToLegacy())});
    try{
    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(source.To(device), 
            target.To(device),
            dataset.get_intrinsics(),
            core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
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

void test_odometry(SubDataset data,
                    t::pipelines::odometry::Method method, 
                    t::pipelines::odometry::MaskMethod m_mehtod,
                    core::Device  device,
                    bool normal){

    auto dataset = Tum_dataset(data);
    auto Trajectory = std::vector<core::Tensor>();

    auto dtype = core::Dtype::Float32;

    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {6,3,1};
    core::Tensor pose = dataset.get_init_pose();
    
    Trajectory.push_back(pose);
    Posegraph<Open3dPosegraphBackend> graph(pose);
    bool sucess;

    for(int i = 0; i< dataset.get_size() -1 ; i++){

        t::pipelines::odometry::OdometryResult result;
        
        if (normal){
            result = Odometry(dataset, i, i+1, method, device, critirias, sucess);
        }
        else{
        switch (m_mehtod)
        {
            case t::pipelines::odometry::MaskMethod::SourceMask:
                result = SOdometry(dataset, i, i+1, method, device, critirias, sucess);
                break;

            case t::pipelines::odometry::MaskMethod::TargetMask:
                result = TOdometry(dataset, i, i+1, method, device, critirias, sucess);
                break;
            
            case t::pipelines::odometry::MaskMethod::CompleteMask:
                result = DOdometry(dataset, i, i+1,  method, device, critirias, sucess);
                break;
            }
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
        
        Trajectory.push_back(pose);
    } 
    std::cout << "Odometry finished" << std::endl;

    auto o3d_posegraph = graph.GetPoseGraph();
    auto intrinsics = dataset.get_intrinsics(core::Device("CPU:0"));
    //auto vgb = integrate(o3d_posegraph, dataset.get_colorfiles_paths(), dataset.get_depthfiles_paths(), intrinsics, 5000.0, 5.0);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((vgb->ExtractPointCloud()).ToLegacy())});

    std::cout << "ATE: " << dataset.ComputeATE(Trajectory, false)<< std::endl;
    std::cout << "RPE: " << dataset.ComputeRPE(Trajectory, 1)<< std::endl;

    graph.Save("graph.json");
    //t::io::WritePointCloud("pointcloud.pcd", vgb->ExtractPointCloud());
};

void test_odometry_optimize(SubDataset data,
                    t::pipelines::odometry::Method method, 
                    t::pipelines::odometry::MaskMethod m_mehtod,
                    core::Device  device,
                    bool normal){

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
            
            if (normal){
                result = Odometry(dataset, i, j, method, device, critirias, sucess);
            }
            else{
            switch (m_mehtod)
            {
                case t::pipelines::odometry::MaskMethod::SourceMask:
                    result = SOdometry(dataset, i, j, method, device, critirias, sucess);
                    break;

                case t::pipelines::odometry::MaskMethod::TargetMask:
                    result = TOdometry(dataset, i, j, method, device, critirias, sucess);
                    break;
                
                case t::pipelines::odometry::MaskMethod::CompleteMask:
                    result = DOdometry(dataset, i, j, method, device, critirias, sucess);
                    break;
                }
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


void test_slam_robust(SubDataset data, core::Device device, SlamMethod method){

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
    t::pipelines::odometry::OdometryResult result1, result2;
    core::Tensor current_T_frame_to_world;
    float weight_threshold;

    for(int i = 1; i< dataset.get_size() -1 ; i++){
        
        weight_threshold = std::min(i*1.0f, 20.0f);
        t::geometry::RGBDMImage source = dataset.get_RGBDMImage(i).To(device);

        source_frame.SetDataFromImage("color", source.color_);
        source_frame.SetDataFromImage("depth", source.depth_);

        source_frame.SetDataFromImage("mask", source.mask_);
        model.SynthesizeModelFrame(ray_cast_frame, depthscale, 0.1, depth_max, 8.0f, true, weight_threshold); 

        current_T_frame_to_world = model.GetCurrentFramePose(); 
        bool succes1, succes2;
        try{ 
            switch (method){
                case SlamMethod::WeightMask:
                    result1 = model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;
                case SlamMethod::Raw:
                    result1 = model.TrackFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;
                case SlamMethod::Masked:
                    result1 =  model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;}
                succes1 = true;
            
        }   
        catch(std::runtime_error){
            std::cout<<"tracking lost at" << dataset.get_timestamp(i) << std::endl;
            result1 = t::pipelines::odometry::OdometryResult();
            succes1 = false;
        }

        result2 = DOdometry(dataset, i, i-1, t::pipelines::odometry::Method::PointToPlane, 
            device, critirias, succes2);
        
        bool similar = AreTransformationsSimilar(result1.transformation_, result2.transformation_, 0.3);
        if (!similar){
            std::cout << i<<std::endl;
        }
        
        core::Tensor trans;
        if (succes1 && similar){
            model.UpdateFramePose(i, current_T_frame_to_world.Matmul(result1.transformation_));
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
            trans = result1.transformation_;
        }
        else if((succes1 && !similar) ||(succes2 && !similar)) {
            model.UpdateFramePose(i, current_T_frame_to_world.Matmul(result2.transformation_));
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
            trans = result2.transformation_;
        }
        else{
            model.UpdateFramePose(i, current_T_frame_to_world);
            trans = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")));
        }

        //std::cout << dataset.get_timestamp(i) << std::endl;
        auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(
            source.depth_.To(device), ray_cast_frame.GetDataAsImage("depth"), 
            dataset.get_intrinsics(core::Device("CPU:0")),trans, 0.1, depthscale, depth_max);
        
        //std::cout << dataset.get_timestamp(i) << std::endl;
        
        graph.AddOdometryEdge(trans, information, i, i+1, false);
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

