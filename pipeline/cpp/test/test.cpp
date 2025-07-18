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

using namespace open3d;
namespace fs = std::filesystem;

int CountFilesDir(fs::path dirPath){

    std::size_t count = 0;

    for(auto& entry: fs::directory_iterator(dirPath)){
        ++count;
    }

    return count;
}

t::pipelines::odometry::OdometryResult DOdometry(Tum_dataset& dataset, int i, t::pipelines::odometry::Method method, core::Device device){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDMImage(i+1);

    auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
            target.To(device),
            dataset.get_intrinsics(device),
            core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
            5000.0F,
            3.0F,
            {10, 5, 3},
            method,
            open3d::t::pipelines::odometry::OdometryLossParams());

    return result;
}

t::pipelines::odometry::OdometryResult TOdometry(Tum_dataset& dataset, int i, t::pipelines::odometry::Method method, core::Device device){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDMImage(i+1);

    auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
            target.To(device), 
            dataset.get_intrinsics(device), 
            core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
            5000.0F,
            3.0F,
            {10, 5, 3},
            method,
            open3d::t::pipelines::odometry::OdometryLossParams());

    return result;
}

t::pipelines::odometry::OdometryResult SOdometry(Tum_dataset& dataset, int i, t::pipelines::odometry::Method method, core::Device device){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDImage(i+1);

    try{
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(device),
                core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
                5000.0F,
                3.0F,
                {10, 5, 3},
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        return result; 
    }
    catch(std::runtime_error){
        return t::pipelines::odometry::OdometryResult();
    }

    
}

t::pipelines::odometry::OdometryResult Odometry(Tum_dataset& dataset, int i, t::pipelines::odometry::Method method, core::Device device){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDImage(i+1);

    auto Pcd = t::geometry::PointCloud::CreateFromDepthImage(source.depth_, dataset.get_intrinsics(device), 
        core::Tensor::Eye(4, core::Float32, ((open3d::core::Device)("CPU:0"))), 1000.0f, 10.0f);
    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((Pcd).ToLegacy())});
    
    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(source.To(device), 
            target.To(device),
            dataset.get_intrinsics(device),
            core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
            5000.0F,
            3.0F,
            {10, 5, 3},
            method,
            open3d::t::pipelines::odometry::OdometryLossParams());

    return result; 
}

void test_masked_odometry(fs::path run_path,
                          t::pipelines::odometry::Method method, 
                          t::pipelines::odometry::MaskMethod m_mehtod,
                          core::Device  device){

    auto dataset = Tum_dataset(run_path);

    auto dtype = core::Dtype::Float32;
    const auto init_trans = core::Tensor::Eye(4,core::Dtype::Float64 ,device);

    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {6,3,1};
    core::Tensor pose = core::Tensor::Eye(4, core::Dtype::Float64 ,device);

    Posegraph<Open3dPosegraphBackend> graph(pose);
    
    for(int i = 0; i< dataset.get_size() -1 ; i++){

        t::pipelines::odometry::OdometryResult result;

        switch (m_mehtod)
        {
        case t::pipelines::odometry::MaskMethod::SourceMask:
            result = SOdometry(dataset, i, method, device);

        case t::pipelines::odometry::MaskMethod::TargetMask:
            result = TOdometry(dataset, i, method, device);
        
        case t::pipelines::odometry::MaskMethod::CompleteMask:
            result = DOdometry(dataset, i, method, device);
        }
        
        t::geometry::RGBDImage source = dataset.get_RGBDImage(i);
        t::geometry::RGBDImage target = dataset.get_RGBDImage(i+1);

        //information muss auch noch geändert werden
        auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(
            source.depth_, target.depth_,
            dataset.get_intrinsics(device), result.transformation_, 0.1);
        
        pose = pose.Matmul(result.transformation_);
        graph.AddOdometryEdge(result.transformation_, information, i, i+1, false);
        graph.AddNode(pose.Inverse());

    } 

    graph.Save(run_path/"graph.json");

    pipelines::registration::PoseGraph o3d_graph;
    io::ReadPoseGraph(run_path/"graph.json", o3d_graph);

    auto intrinsics = dataset.get_intrinsics(device);
    auto vgb = integrate(o3d_graph, dataset.get_colorfiles(), dataset.get_depthfiles(), intrinsics);
    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((vgb->ExtractPointCloud()).ToLegacy())});

};
