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

Tum_dataset::Tum_dataset(SubDataset dataset){

    fs::path rootdir = dataset_paths[dataset];

    auto color_dir = rootdir / "rgb_aligned";
    auto depth_dir = rootdir / "depth_aligned";
    auto mask_dir = rootdir/ "mask";

    if (!(fs::exists(color_dir)) || !(fs::exists(depth_dir)) || !(fs::exists(mask_dir))){
        std::cout << "Invalid folder structure" << std::endl;
    }

    root_dir = rootdir;

    color_file_paths = GetFilesDir(color_dir);
    depth_file_paths = GetFilesDir(depth_dir);
    mask_file_paths = GetFilesDir(mask_dir);

    color_files = std::vector<t::geometry::Image>();
    depth_files = std::vector<t::geometry::Image>();
    mask_files = std::vector<t::geometry::Image>();

    t::geometry::Image d_image;
    t::geometry::Image c_image;
    for (int i = 0; i < color_file_paths.size(); i ++){

        t::io::ReadImageFromPNG((depth_file_paths[i]).string(), d_image);
        t::io::ReadImageFromPNG((color_file_paths[i]).string(), c_image);
        auto mask = core::Tensor::Load((mask_file_paths[i]).string());
        auto mask_image = t::geometry::Image(mask);

        color_files.push_back(c_image);
        depth_files.push_back(d_image);
        mask_files.push_back(mask_image);
    }

    size = mask_files.size();

    intrinsics = core::Tensor::Init<double>({
    535.4, 0.0,   320.1, 
    0.0,   539.2, 247.6, 
    0.0,   0.0,   1.0}).Reshape({3, 3});


    GroundTruth = this->LoadTrajectoryToTensors((root_dir/"groundtruth_aligned.txt").string()); 
}

t::geometry::Image Tum_dataset::get_color(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
    
    return color_files[i];
    }
}

t::geometry::Image Tum_dataset::get_depth(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        return depth_files[i];
    }
}

t::geometry::Image Tum_dataset::get_mask(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        return mask_files[i];
    }
}

t::geometry::RGBDImage Tum_dataset::get_RGBDImage(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        
        return t::geometry::RGBDImage(color_files[i], depth_files[i]);
    }
}

t::geometry::RGBDMImage Tum_dataset::get_RGBDMImage(int i){

    if (i > size){
        throw std::runtime_error("index out of range");
    }
    else{
        return t::geometry::RGBDMImage(color_files[i], depth_files[i], mask_files[i]);
    }
}

core::Tensor SE3FromQuatAndTrans(double tx, double ty, double tz,
                                  double qx, double qy, double qz, double qw,
                                  const core::Device& device = core::Device("CPU:0")) {
    Eigen::Quaterniond q(qw, qx, qy, qz); 
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = Eigen::Vector3d(tx, ty, tz);

    return core::eigen_converter::EigenMatrixToTensor(T);
}

std::vector<core::Tensor> Tum_dataset::LoadTrajectoryToTensors(const std::string& file_path,
                                                               const core::Device& device) {
    std::ifstream file(file_path);
    std::vector<core::Tensor> transforms;

    if (!file.is_open()) {
        utility::LogError("Cannot open file: {}", file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        double t, tx, ty, tz, qx, qy, qz, qw;
        if (!(ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            continue;  // skip malformed lines
        }

        core::Tensor T = SE3FromQuatAndTrans(tx, ty, tz, qx, qy, qz, qw, device);
        transforms.push_back(T);
    }

    return transforms;
}

Eigen::Matrix4d ComputeUmeyamaAlignment(const std::vector<Eigen::Vector3d>& gt,
                                        const std::vector<Eigen::Vector3d>& est) {
    assert(gt.size() == est.size());
    const size_t N = gt.size();

    Eigen::MatrixXd src(3, N), dst(3, N);
    for (size_t i = 0; i < N; ++i) {
        src.col(i) = est[i];
        dst.col(i) = gt[i];
    }

    // Eigen's Umeyama implementation: scale = false
    Eigen::Matrix4d T = Eigen::umeyama(src, dst, false);
    return T;
}

double ComputeTensorATE(const std::vector<core::Tensor>& gt_poses,
                  const std::vector<core::Tensor>& est_poses, 
                  bool align) {

    assert(gt_poses.size() == est_poses.size());
    const size_t N = gt_poses.size();

    std::vector<Eigen::Vector3d> gt_trans, est_trans;
    for (size_t i = 0; i < N; ++i) {
        auto gt_pose_eigen = core::eigen_converter::TensorToEigenMatrixXd(gt_poses[i]);
        auto est_pose_eigen = core::eigen_converter::TensorToEigenMatrixXd(est_poses[i]);

        gt_trans.push_back(gt_pose_eigen.block<3,1>(0,3));
        est_trans.push_back(est_pose_eigen.block<3,1>(0,3));
    }

    double error_sum = 0.0;
    if (align){
        Eigen::Matrix4d T_align = ComputeUmeyamaAlignment(gt_trans, est_trans);
        for (size_t i = 0; i < N; ++i) {
            Eigen::Vector4d p_est;
            p_est.head<3>() = est_trans[i];
            p_est[3] = 1.0;

            Eigen::Vector3d aligned = (T_align * p_est).head<3>();
            double error = (aligned - gt_trans[i]).norm();
            error_sum += error * error;}
    }else{
        for (size_t i = 0; i < N; ++i) {
            
            double error = (est_trans[i] - gt_trans[i]).norm();
            error_sum += error * error;}
    }

    return std::sqrt(error_sum / N);
}


double ComputeTensorRPE(const std::vector<open3d::core::Tensor>& gt_poses,
                        const std::vector<open3d::core::Tensor>& est_poses,
                        int delta = 1) {

    assert(gt_poses.size() == est_poses.size());
    assert(delta > 0 && gt_poses.size() > delta);

    int N = static_cast<int>(gt_poses.size());
    double error_sum = 0.0;
    int count = 0;

    for (int i = 0; i + delta < N; ++i) {
        const auto& Ti_gt  = gt_poses[i];
        const auto& Tip_gt = gt_poses[i + delta];
        const auto& Ti_est  = est_poses[i];
        const auto& Tip_est = est_poses[i + delta];

        // Relative transformations
        core::Tensor rel_gt  = Ti_gt.Inverse().Matmul(Tip_gt);
        core::Tensor rel_est = Ti_est.Inverse().Matmul(Tip_est);

        // RPE transform: error = inverse(gt_rel) * est_rel
        core::Tensor error = rel_gt.Inverse().Matmul(rel_est);

        // Extract translation vector (top-right 3x1 block)
        core::Tensor trans = error.Slice(0, 0, 3).Slice(1, 3, 4);  // shape: (3, 1)

        double norm = trans.Mul(trans).Sum({0}).Sqrt().Item<double>();
        //std::cout<< norm <<std::endl;
        error_sum += norm;
        count++;
    }

    return std::sqrt(error_sum / count);
}

double Tum_dataset::ComputeATE(std::vector<core::Tensor> Trajectory, bool align){

    return ComputeTensorATE(GroundTruth, Trajectory, align);
}

double Tum_dataset::ComputeRPE(std::vector<core::Tensor> Trajectory, int delta){

    return ComputeTensorRPE(GroundTruth, Trajectory, delta);
}