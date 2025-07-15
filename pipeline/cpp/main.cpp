#include <open3d/core/Tensor.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/io/IJsonConvertibleIO.h>


#include "test/test.h"
#include <filesystem>

using namespace open3d;
namespace fs =  std::filesystem;

int main(){
    
    //open3d::utility::Logger::GetInstance().SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);
    fs::path core("/home/hauke/code/Beachlor");

    camera::PinholeCameraIntrinsic intrinsics;
    io::ReadIJsonConvertible(core / "data/intrinsics/intrinsics.json", intrinsics);
    auto intrinsics_matrix = core::eigen_converter::EigenMatrixToTensor(intrinsics.intrinsic_matrix_);

    fs::path run_path(core / "data/test");
    test_masked_odometry(run_path, intrinsics_matrix);

    return 0;
};

