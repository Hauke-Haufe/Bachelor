#include"open3d/Open3D.h"


#include <filesystem>
#include <algorithm>
#include <iostream>

using namespace open3d;
namespace fs = std::filesystem;

class Tum_dataset{
    public:
    Tum_dataset(fs::path root_dir);

    t::geometry::Image get_mask(int i);
    t::geometry::Image get_depth(int i);
    t::geometry::Image get_color(int i);

    t::geometry::RGBDImage get_RGBDImage(int i);
    t::geometry::RGBDMImage get_RGBDMImage(int i);

    inline size_t get_size(){
        return size;
    };

    inline std::vector<fs::path> get_colorfiles(){
        return color_files;
    };

    inline std::vector<fs::path> get_depthfiles(){
        return depth_files;
    };

    inline std::vector<fs::path> get_maskfiles() {
        return mask_files;
    };

    inline core::Tensor get_intrinsics(core::Device device){
        return intrinsics.To(device);
    };
    
    private:
    fs::path root_dir;
    size_t size;

    std::vector<fs::path> color_files;
    std::vector<fs::path> depth_files;
    std::vector<fs::path> mask_files;

    core::Tensor intrinsics;
};