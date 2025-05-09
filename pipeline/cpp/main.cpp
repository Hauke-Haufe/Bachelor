#include <open3d/Open3D.h>
//#include <chrono>
//#include "odometry.cpp"
#include "SemanticVBG.h"

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

template <typename Derived>
core::Tensor EigentoTensorF64i(Eigen::MatrixBase<Derived> matrix){

    constexpr int rows = Derived::RowsAtCompileTime;
    constexpr int  cols = Derived::ColsAtCompileTime;
    core::SizeVector size({rows, cols});

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_row = matrix;

    std::cout<<mat_row<<std::endl;
    std::cin;

    return core::Tensor(
        static_cast<double*>(mat_row.data()), 
        core::Dtype::Float64, size
    );
    
}

template <typename Derived>
core::Tensor EigentoTensorF64(Eigen::MatrixBase<Derived>& matrix){
    
    constexpr int rows = Derived::RowsAtCompileTime;
    constexpr int  cols = Derived::ColsAtCompileTime;
    core::SizeVector size({rows, cols});

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_row = matrix;
    core::Tensor tensor(size, core::Dtype::Float64);

    for (int i = 0; i< rows; ++i){
        for (int j = 0 ; j< cols; ++j){
            tensor[i][j] = mat_row(i,j);
        }
    }
    return tensor;

    
}

int main(){
    
    //utility::VerbosityContextManager contex(utility::VerbosityLevel::Debug);
    //contex.Enter();
    auto color_image = std::make_shared<t::geometry::Image>();
    auto depth_image = std::make_shared<t::geometry::Image>();

    camera::PinholeCameraIntrinsic intrinsics;
    io::ReadIJsonConvertible("/home/nb-messen-07/Desktop/SpatialMapping/data/intrinsics.json", intrinsics);

    auto intrinsics_matrix = EigentoTensorF64(intrinsics.intrinsic_matrix_);
    core::Device device("CUDA:0");
    core::Tensor extrinsic = core::Tensor::Eye(4, core::Dtype::Float32, device);


    //auto images = load_images("/home/nb-messen-07/Desktop/SpatialMapping/data/images", 10,100, 1);
    //auto posegraph = Multiway_registration(images, intrinsics_matrix); 
    //auto posegraph_opt = optimize_posegraph(posegraph);
    //auto vgb = integrate(posegraph_opt, intrinsics_matrix, "/home/nb-messen-07/Desktop/SpatialMapping/data/images" );
    
    SemanticVBG vgb(0.01, 16, 10000, device);
    auto block_coords = vgb.GetUniqueBlockCoordinates(*depth_image, intrinsics_matrix, extrinsic);
    //vbg.Integrate()
    
    return 0;
};

