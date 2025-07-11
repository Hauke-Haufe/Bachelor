#include <vector>
#include <filesystem>

#include <open3d/t/geometry/VoxelBlockGrid.h>
#include <open3d/pipelines/registration/PoseGraph.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/io/ImageIO.h>
#include <open3d/core/EigenConverter.h>

std::unique_ptr<open3d::t::geometry::VoxelBlockGrid> create_vgb();

std::unique_ptr<open3d::t::geometry::VoxelBlockGrid> integrate(
    open3d::pipelines::registration::PoseGraph& Posegraph,
    const std::vector<std::filesystem::path>& color_images,
    const std::vector<std::filesystem::path>& depth_images, 
    open3d::core::Tensor& instrinsics);