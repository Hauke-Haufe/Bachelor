#include <open3d/Open3D.h>
#include <filesystem>
#include <sstream>
#include <vector>
#include <algorithm>



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

void test_masked_odometry(fs::path run_path){

    fs::path image_dir = run_path / "images";
    fs::path mask_dir = run_path / "mask";

    auto color_images = GetFilesDir(image_dir);
    auto mask_images = GetFilesDir(mask_dir);

    auto s_color_image = std::make_shared<t::geometry::Image>();
    auto s_depth_image = std::make_shared<t::geometry::Image>();
    auto t_color_image = std::make_shared<t::geometry::Image>();
    auto d_depthImage = std::make_shared<t::geometry::Image>();
    auto source_mask = std::make_shared<t::geometry::Image>();

    for(int i = 0; i< CountFilesDir(run_path)-1; i++){

        t::io::ReadImageFromPNG((image_dir / color_images[i]).string(), *s_color_image);
        t::io::ReadImageFromPNG((image_dir / color_images[i+1]).string(), *t_color_image);

        t::io::ReadImageFromPNG((mask_dir / mask_images[i]).string(), *source_mask);
       
        auto result = t::pipelines::odometry::RGBDMaskOdometryMultiScaleHybrid(
            
        );
    } 

};