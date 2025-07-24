#pragma once

#include"open3d/Open3D.h"
#include <filesystem>
#include <algorithm>
#include <iostream>

using namespace open3d;
namespace fs = std::filesystem;

enum class SubDataset{
    static_xyz,
    walking_xyz,
    walking_halfsphere,
    walking_rpy,
    walking_static
};


//class for handeling a dataset from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
//Images need to be aligned use for this the align_TUMDataset.py script
class Tum_dataset{
    public:

    std::unordered_map<SubDataset, std::string> dataset_paths = {
        {SubDataset::static_xyz,        "data/rgbd_dataset_freiburg1_xyz"},
        {SubDataset::walking_xyz,       "data/rgbd_dataset_freiburg3_walking_xyz"},
        {SubDataset::walking_halfsphere,"data/rgbd_dataset_freiburg3_walking_halfsphere"},
        {SubDataset::walking_rpy,       "data/rgbd_dataset_freiburg3_walking_rpy"},
        {SubDataset::walking_static,    "data/rgbd_dataset_freiburg3_walking_static" }
    };

    Tum_dataset(SubDataset dataset);

    t::geometry::Image get_mask(int i);
    t::geometry::Image get_depth(int i);
    t::geometry::Image get_color(int i);

    t::geometry::RGBDImage get_RGBDImage(int i);
    t::geometry::RGBDMImage get_RGBDMImage(int i);

    core::Tensor get_init_pose(){
        return GroundTruth[0];
    }

    inline size_t get_size(){
        return size;
    };

    inline std::vector<fs::path> get_colorfiles_paths(){
        return color_file_paths;
    };

    inline std::vector<fs::path> get_depthfiles_paths(){
        return depth_file_paths;
    };

    inline std::vector<fs::path> get_maskfiles_paths() {
        return mask_file_paths;
    };

    inline core::Tensor get_intrinsics(core::Device device = core::Device("CPU:0")){
        return intrinsics.To(device);
    };

    inline std::string get_timestamp(int i){
        if (i > size){
            throw std::runtime_error("index out of range");
        }
        return color_file_paths[i];
    }

    double ComputeATE(std::vector<core::Tensor> Trajectory, bool align);

    double ComputeRPE(std::vector<core::Tensor> Trajectory, int delta);
    
    private:

    std::vector<core::Tensor> LoadTrajectoryToTensors(const std::string& file_path,
                                const core::Device& device = core::Device("CPU:0"));

    fs::path root_dir;
    size_t size;
    core::Tensor intrinsics;
    
    std::vector<core::Tensor> GroundTruth;

    std::vector<fs::path> color_file_paths;
    std::vector<fs::path> depth_file_paths;
    std::vector<fs::path> mask_file_paths;


    std::vector<t::geometry::Image> color_files;
    std::vector<t::geometry::Image> depth_files;
    std::vector<t::geometry::Image> mask_files;
};