#include <open3d/core/Tensor.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/io/IJsonConvertibleIO.h>
#include <open3d/t/pipelines/odometry/RGBDMOdometry.h>
#include <open3d/t/pipelines/odometry/RGBDOdometry.h>


#include "test/test.h"
#include "test/test_dataset.h"
#include <filesystem>

using namespace open3d;
namespace fs =  std::filesystem;

int main(){
    
    //open3d::utility::Logger::GetInstance().SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);
    core::Device device(core::Device::DeviceType::CUDA, 0);

    test_slam(SubDataset::walking_xyz, 
        device,
        SlamMethod::Raw);
        
    
    /*auto t = core::Tensor::Load("/home/hauke/code/Beachlor/data/test_dataset/rgbd_dataset_freiburg3_walking_xyz/mask/1341846313.592026.png.npy");
    auto t_new = t.To(core::Dtype::Bool).To(core::Dtype::Float32);
    t::io::WriteNpy("t.npy", t::geometry::Image(t_new).FilterGaussian(45, 15).AsTensor());*/

    return 0;
};

