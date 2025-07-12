#include "test.h"
#include "../pose_graph.h"
#include "../Integration.h"

//#include <open3d/Open3D.h>
#include <open3d/t/pipelines/odometry/RGBDOdometry.h>
#include <open3d/core/Device.h>
#include <open3d/core/Dtype.h>
#include <open3d/t/io/ImageIO.h>
#include <open3d/pipelines/registration/GlobalOptimization.h>

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

std::vector<fs::path> GetFilesDir(fs::path dirPath){

    std::vector<fs::path> files;
    for(auto& entry: fs::directory_iterator(dirPath)){
        files.emplace_back(entry.path());
    }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){return a < b;});

    return files;
} 


void test_masked_odometry(fs::path run_path, core::Tensor intrinsic_matrix){

    fs::path image_dir = run_path / "color";
    fs::path mask_dir = run_path / "masks";
    fs::path depth_dir = run_path / "depth";

    core::Device device(core::Device::DeviceType::CPU, 0);
    core::Device cuda_device(core::Device::DeviceType::CUDA, 0);

    auto dtype = core::Dtype::Float32;
    const auto init_trans = core::Tensor::Eye(4,core::Dtype::Float64 ,device);

    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {6,3,1};

    auto color_images = GetFilesDir(image_dir);
    auto mask_images = GetFilesDir(mask_dir);
    auto depth_images = GetFilesDir(depth_dir);

    auto s_color_image = std::make_shared<t::geometry::Image>();
    auto t_color_image = std::make_shared<t::geometry::Image>();
    auto t_depth_image = std::make_shared<t::geometry::Image>();
    auto s_depth_image = std::make_shared<t::geometry::Image>();
    
    auto temp_mask = std::make_shared<t::geometry::Image>();
    auto source_mask = temp_mask->To(core::Dtype::Bool);
    auto target_mask = temp_mask->To(core::Dtype::Bool);
    
    auto pose = core::Tensor::Eye(4,core::Dtype::Float64 ,device);
    auto m_pose = core::Tensor::Eye(4,core::Dtype::Float64 ,device);

    Posegraph<Open3dPosegraphBackend> m_graph(m_pose);
    Posegraph<Open3dPosegraphBackend> graph(pose);
    
    for(int i = 0; i< CountFilesDir(image_dir)-1; i++){
        
        t::io::ReadImageFromPNG((color_images[i]).string(), *s_color_image);
        t::io::ReadImageFromPNG((color_images[i+1]).string(), *t_color_image);
        t::io::ReadImageFromPNG((depth_images[i]).string(), *s_depth_image);
        t::io::ReadImageFromPNG((depth_images[i+1]).string(), *t_depth_image);

        t::io::ReadImageFromPNG((mask_images[i]).string(), source_mask);
        t::io::ReadImageFromPNG((mask_images[i+1]).string(), target_mask);

        auto source = t::geometry::RGBDMImage(*s_color_image, *s_depth_image, source_mask);//.To(cuda_device);
        auto target = t::geometry::RGBDMImage(*t_color_image, *t_depth_image, target_mask);//.To(cuda_device);
        auto s = target.GetDevice();

        auto m_result = t::pipelines::odometry::RGBDMOdometryMultiScale(
            source, target,intrinsic_matrix, init_trans, 
            1000.0f, 3.0f, critirias, t::pipelines::odometry::OdometryLossParams() 
        );

        auto result = t::pipelines::odometry::RGBDOdometryMultiScale(
            t::geometry::RGBDImage(source.color_, source.depth_), t::geometry::RGBDImage(target.color_, target.depth_), 
            intrinsic_matrix, init_trans,1000.0f, 3.0f, critirias, 
            t::pipelines::odometry::Method::Hybrid, t::pipelines::odometry::OdometryLossParams());

        //information muss auch noch geändert werden
        auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(*s_depth_image,
        *t_depth_image, intrinsic_matrix, result.transformation_, 0.1);

        auto m_information = t::pipelines::odometry::ComputeOdometryInformationMatrix(*s_depth_image,
        *t_depth_image, intrinsic_matrix, m_result.transformation_, 0.1);
        
        pose = pose.Matmul(result.transformation_);
        m_pose = m_pose.Matmul(m_result.transformation_);
        
        graph.AddOdometryEdge(result.transformation_, information, i, i+1, false);
        m_graph.AddOdometryEdge(m_result.transformation_, m_information, i, i+1, false);

        graph.AddNode(pose.Inverse());
        m_graph.AddNode(m_pose.Inverse());
    } 

    graph.Save(run_path/"graph.json");
    m_graph.Save(run_path/"masked_graph.json");

    pipelines::registration::PoseGraph o3d_mgraph;
    pipelines::registration::PoseGraph o3d_graph;

    io::ReadPoseGraph(run_path/"graph.json", o3d_graph);
    io::ReadPoseGraph(run_path/"masked_graph.json", o3d_mgraph);

    std::vector<t::geometry::PointCloud> DrawObjects;

    auto m_vgb = integrate(o3d_mgraph, color_images, depth_images, intrinsic_matrix);
    auto vgb = integrate(o3d_graph, color_images, depth_images, intrinsic_matrix);

    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((vgb->ExtractPointCloud()).ToLegacy())});
    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((m_vgb->ExtractPointCloud()).ToLegacy())});
};