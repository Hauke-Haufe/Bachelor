#include <open3d/Open3D.h>
#include <chrono>
#include "odometry.cpp"

using namespace open3d;

//assumes path/color/images{i}.png path/depth/images{i}.png 
std::vector<std::shared_ptr<t::geometry::RGBDImage>> load_images(std::string path, int sid, int eid, int freq){

    std::vector<std::shared_ptr<t::geometry::RGBDImage>> images;
    assert(sid < eid);

    images.reserve(eid-sid);
    
    t::geometry::Image depth_image;
    t::geometry::Image color_image;

    for(int i = sid; sid < eid; sid = sid +freq ){

        std::stringstream color_path_ss;
        std::stringstream depth_path_ss;
        color_path_ss << path << "/color/image" << i << ".png";
        depth_path_ss << path << "/depth/image" << i << ".png";

        std::string color_path = color_path_ss.str();
        std::string depth_path = depth_path_ss.str();

        t::io::ReadImage(color_path, color_image);
        t::io::ReadImage(depth_path, depth_image);

        auto image = std::make_shared<t::geometry::RGBDImage>(color_image, depth_image);
        
        images.push_back(image);
    }

    return images;
}

int main(){
    
    //utility::VerbosityContextManager contex(utility::VerbosityLevel::Debug);
    //contex.Enter();
    auto color_image = std::make_shared<t::geometry::Image>();
    auto depth_image = std::make_shared<t::geometry::Image>();

    camera::PinholeCameraIntrinsic intrinsics;
    io::ReadIJsonConvertible("/home/nb-messen-07/Desktop/SpatialMapping/data/intrinsics.json", intrinsics);

    auto intrinsics_matrix = EigentoTensorF64(intrinsics.intrinsic_matrix_);


    auto images = load_images("/home/nb-messen-07/Desktop/SpatialMapping/data/images", 10,100, 1);
    auto posegraph = Multiway_registration(images, intrinsics_matrix); 
    auto posegraph_opt = optimize_posegraph(posegraph);
    auto vgb = integrate(posegraph_opt, intrinsics_matrix, "/home/nb-messen-07/Desktop/SpatialMapping/data/images" );

    
    return 0;
};

