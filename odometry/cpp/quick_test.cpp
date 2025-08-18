#include <open3d/Open3D.h>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;
using namespace open3d;
using core::Tensor;

enum class SlamMethod { Raw, Maskout, WeightMaskout };

// --- Helpers ----------------------------------------------------------------

static std::vector<fs::path> ListSorted(const fs::path& dir) {
    std::vector<fs::path> files;
    for (auto &e : fs::directory_iterator(dir)) {
        if (e.is_regular_file()) files.emplace_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Read tensor image via Open3D (tensor API)
static t::geometry::Image ReadTImage(const fs::path& p) {
    t::geometry::Image img;
    if (!t::io::ReadImage(p.string(), img)) {
        throw std::runtime_error("Failed to read image: " + p.string());
    }
    return img;
}

static t::geometry::Image ReadTImageMask(const fs::path& p) {
    t::geometry::Image img;
    
    auto mask = core::Tensor::Load((p).string());
    auto mask_image = t::geometry::Image(mask.To(core::Bool));
    return mask_image;
}

// --- Core function -----------------------------------------------------------

void BuildAndShowPCDFromDirs(const fs::path& base,
                             const core::Tensor& intrinsics,
                             const core::Device& device,
                             SlamMethod method,
                             t::pipelines::odometry::Method o_method) {
    // Directories: data/images/{color,depth,mask}
    const fs::path color_dir = base / "images" / "color";
    const fs::path depth_dir = base / "images" / "depth";
    const fs::path mask_dir  = base / "images" / "mask";

    auto color_files = ListSorted(color_dir);
    auto depth_files = ListSorted(depth_dir);
    auto mask_files  = ListSorted(mask_dir);

    if (color_files.empty() || depth_files.empty() || mask_files.empty())
        throw std::runtime_error("One or more image folders are empty.");

    if (!(color_files.size() == depth_files.size() && color_files.size() == mask_files.size()))
        throw std::runtime_error("Mismatched counts between color/depth/mask.");

    // Read the first frame to get size
    t::geometry::Image first_color = ReadTImage(color_files[20]);
    t::geometry::Image first_depth = ReadTImage(depth_files[20]);
    t::geometry::Image first_mask  = ReadTImageMask(mask_files[20]);

    const int h = first_color.GetRows();
    const int w = first_color.GetCols();

    // SLAM / TSDF model (adjust parameters if needed)
    const float depth_scale = 1000.0f;  // depth in mm
    const float depth_max   = 3.0f;

    t::pipelines::slam::Model model(/*voxel_size=*/0.01, /*block_resolution=*/16,
                                    /*block_count=*/1000);

    // Frames
    t::pipelines::slam::Frame ray_cast_frame(h, w, intrinsics, device);
    t::pipelines::slam::Frame source_frame (h, w, intrinsics, device);

    // Initialize pose with identity
    core::Device d("CPU:0");
    Tensor init_pose = core::Tensor::Eye(4, core::Float64, d );
    model.UpdateFramePose(0, init_pose);

    // --- Integrate first frame ------------------------------------------------
    source_frame.SetDataFromImage("color", first_color.To(device));
    source_frame.SetDataFromImage("depth", first_depth.To(device));
    source_frame.SetDataFromImage("mask" , first_mask.To(device));

    switch (method) {
        case SlamMethod::Raw:
            model.Integrate(source_frame, depth_scale, depth_max);
            break;
        case SlamMethod::Maskout:
            model.MaskedIntegrate(source_frame, depth_scale, depth_max);
            break;
        case SlamMethod::WeightMaskout:
            model.WeightMaskedIntegrate(source_frame, depth_scale, depth_max);
            break;
    }

    // --- Process the rest -----------------------------------------------------
    for (size_t i = 20; i < 300; ++i) {
        double weight_threshold = std::min(i*1.0f -20, 4.0f);
        t::geometry::Image color = ReadTImage(color_files[i]).To(device);
        t::geometry::Image depth = ReadTImage(depth_files[i]).To(device);
        t::geometry::Image mask  = ReadTImageMask(mask_files[i]).To(device);

        source_frame.SetDataFromImage("color", color);
        source_frame.SetDataFromImage("depth", depth);
        source_frame.SetDataFromImage("mask" , mask);

        // Raycast current model to synthesize target frame
        model.SynthesizeModelFrame(ray_cast_frame,
                                   depth_scale,
                                   /*depth_min=*/0.1f,
                                   depth_max,
                                   /*sdf_trunc=*/8.0f,
                                   /*color_type=*/true,
                                   weight_threshold);

        // Track (estimate relative transform)
        t::pipelines::odometry::OdometryResult result;
        try {
            switch (method) {
                case SlamMethod::Raw:
                    result = model.TrackFrameToModel(source_frame, ray_cast_frame,
                                                     depth_scale, depth_max,
                                                     /*sigma=*/0.07, o_method,
                                                     /*criteria per level*/ {10,10,10});
                    break;
                case SlamMethod::Maskout:
                    result = model.TrackMaskedFrameToModel(source_frame, ray_cast_frame,
                                                           depth_scale, depth_max,
                                                           /*sigma=*/0.07, o_method,
                                                           /*criteria*/ {10,10,10});
                    break;
                case SlamMethod::WeightMaskout:
                    result = model.TrackMaskedFrameToModel(source_frame, ray_cast_frame,
                                                           depth_scale, depth_max,
                                                           /*sigma=*/0.07,
                                                           t::pipelines::odometry::Method::PointToPlane,
                                                           /*criteria*/ {10,10,10});
                    break;
            }
        } catch (const std::runtime_error&) {
            // Tracking failed: keep previous pose (no update) and continue
            model.UpdateFramePose((int)i, model.GetCurrentFramePose());
            continue;
        }

        // Update global pose: T_wf := T_wf * T_f_prev_to_f_curr
        Tensor T_wf_prev = model.GetCurrentFramePose();
        model.UpdateFramePose((int)i, T_wf_prev.Matmul(result.transformation_));

        // Integrate
        switch (method) {
            case SlamMethod::Raw:
                model.Integrate(source_frame, depth_scale, depth_max);
                break;
            case SlamMethod::Maskout:
                model.MaskedIntegrate(source_frame, depth_scale, depth_max);
                break;
            case SlamMethod::WeightMaskout:
                model.WeightMaskedIntegrate(source_frame, depth_scale, depth_max,
                                            /*sdf_trunc=*/8.0, /*max_weight=*/21, /*weight_gamma=*/15);
                break;
        }
    }

    // --- Extract & visualize --------------------------------------------------
    auto pcd = model.ExtractPointCloud();
    t::io::WritePointCloud("r.pcd", pcd);
    visualization::DrawGeometries(
        {std::make_shared<geometry::PointCloud>(pcd.ToLegacy())});
}


int main(){
    camera::PinholeCameraIntrinsic intrinsics;
    io::ReadIJsonConvertibleFromJSON("/home/hauke/code/Beachlor/data/intrinsics/intrinsics.json", intrinsics);
    auto intrinsics_matrix = core::eigen_converter::EigenMatrixToTensor(intrinsics.intrinsic_matrix_);
    core::Device device("CUDA:0");
    BuildAndShowPCDFromDirs("/home/hauke/code/Beachlor/data", intrinsics_matrix, device, SlamMethod::Raw, 
    t::pipelines::odometry::Method::PointToPlane);

}