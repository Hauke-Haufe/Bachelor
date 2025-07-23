#include <filesystem>
#include <open3d/core/Tensor.h>
#include "open3d/core/EigenConverter.h"
#include "open3d/t/pipelines/odometry/RGBDMOdometry.h"
#include "test_util.h"

using namespace open3d;
namespace fs = std::filesystem;


inline int CountFilesDir(std::filesystem::path dirPath){

    std::size_t count = 0;

    for(auto& entry: std::filesystem::directory_iterator(dirPath)){
        ++count;
    }

    return count;
}

bool IsLargeRotation(const open3d::core::Tensor& trans, double angle_threshold_deg ) {
    using namespace open3d;

    // Convert to Eigen
    Eigen::Matrix4d rel = core::eigen_converter::TensorToEigenMatrixXd(
                         trans.To(core::Device("CPU:0")));

    // Extract rotation matrix
    Eigen::Matrix3d R = rel.block<3, 3>(0, 0);

    // Compute rotation angle from trace(R) = 1 + 2cos(theta)
    double trace_R = R.trace();
    double cos_theta = std::clamp((trace_R - 1.0) / 2.0, -1.0, 1.0);  // Numerical safety
    double angle_rad = std::acos(cos_theta);
    double angle_deg = angle_rad * 180.0 / M_PI;

    return angle_deg > angle_threshold_deg;
}


bool IsLargeTranslation(const open3d::core::Tensor& transformation, float threshold) {  // adjust threshold as needed
    // Assume shape [4, 4] and Float32 or Float64
    if (transformation.GetShape() != open3d::core::SizeVector{4, 4}) {
        throw std::runtime_error("Transformation must be a 4x4 matrix.");
    }

    // Copy to CPU for access
    auto T_cpu = transformation.To(open3d::core::Device("CPU:0"));

    // Extract translation components (last column)
    float tx = T_cpu[0][3].Item<double>();
    float ty = T_cpu[1][3].Item<double>();
    float tz = T_cpu[2][3].Item<double>();

    float translation_norm = std::sqrt(tx * tx + ty * ty + tz * tz);

    if  (translation_norm > threshold){
        float t = 1;
    }

    return translation_norm > threshold;
}

bool AreTransformationsSimilar(const open3d::core::Tensor& transform1, const open3d::core::Tensor& transform2, float threshold) {
    // Check if the shapes of the tensors are identical (4x4 matrices)
    if (transform1.GetShape() != transform2.GetShape()) {
        std::cerr << "Transformations have different shapes!" << std::endl;
        return false;
    }

    // Flatten tensors to compare each element
    auto tensor1_data = transform1.Contiguous();
    auto tensor2_data = transform2.Contiguous();

    // Compare the elements of the two tensors
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float diff = std::fabs(transform1[row][ col].Item<double>() - transform2[row][ col].Item<double>());
            if (diff > threshold) {
                return false; // If any difference is larger than the threshold, they are not similar
            }
        }
    }

    return true; 
}