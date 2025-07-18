#include"open3d/Open3D.h"
#include"test_dataset.h"

#include <filesystem>
#include <algorithm>
#include <iostream>

using namespace open3d;
namespace fs = std::filesystem;

std::vector<fs::path> GetFilesDir(fs::path dirPath){

    std::vector<fs::path> files;
    for(auto& entry: fs::directory_iterator(dirPath)){
        files.emplace_back(entry.path());
    }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){return a < b;});

    return files;
} 

Tum_dataset::Tum_dataset(fs::path rootdir){

    auto color_dir = rootdir / "rgb";
    auto depth_dir = rootdir / "depth";
    auto mask_dir = rootdir/ "mask";

    if (!(fs::exists(color_dir)) || !(fs::exists(depth_dir)) || !(fs::exists(mask_dir))){
        std::cout << "Invalid folder structure" << std::endl;
    }

    root_dir = rootdir;

    color_files = GetFilesDir(color_dir);
    depth_files = GetFilesDir(depth_dir);
    mask_files = GetFilesDir(mask_dir);

    size = mask_files.size();

    intrinsics = core::Tensor::Init<double>({
    535.4, 0.0,   320.1, 
    0.0,   539.2, 247.6, 
    0.0,   0.0,   1.0}).Reshape({3, 3});
}

t::geometry::Image Tum_dataset::get_color(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
    t::geometry::Image image;
    return t::io::ReadImageFromPNG((color_files[i]).string(), image);
    }
}

t::geometry::Image Tum_dataset::get_depth(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        t::geometry::Image image;
        return t::io::ReadImageFromPNG((depth_files[i]).string(), image);
    }
}

t::geometry::Image Tum_dataset::get_mask(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        t::geometry::Image image;
        return t::io::ReadImageFromPNG((mask_files[i]).string(), image);
    }
}

t::geometry::RGBDImage Tum_dataset::get_RGBDImage(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        t::geometry::Image d_image;
        t::geometry::Image c_image;

        t::io::ReadImageFromPNG((depth_files[i]).string(), d_image);
        t::io::ReadImageFromPNG((color_files[i]).string(), c_image);
        
        return t::geometry::RGBDImage(c_image, d_image);
    }
}

t::geometry::RGBDMImage Tum_dataset::get_RGBDMImage(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        t::geometry::Image d_image;
        t::geometry::Image c_image;

        t::io::ReadImageFromPNG((depth_files[i]).string(), d_image);
        t::io::ReadImageFromPNG((color_files[i]).string(), c_image);
        std::cout << (color_files[i]).string() << std::endl;
        auto mask = core::Tensor::Load((mask_files[i]).string());
        auto m_image = t::geometry::Image(mask);
        
        return t::geometry::RGBDMImage(c_image, d_image, m_image);
    }
}