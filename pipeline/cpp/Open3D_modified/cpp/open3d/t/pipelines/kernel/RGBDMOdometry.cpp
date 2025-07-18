#include "open3d/t/pipelines/kernel/RGBDOdometry.h"

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/pipelines/kernel/RGBDMOdometryImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ComputeDMaskOdometryResultHybrid(const core::Tensor &source_depth,
                                 const core::Tensor &target_depth,
                                 const core::Tensor &source_intensity,
                                 const core::Tensor &target_intensity,
                                 const core::Tensor &target_depth_dx,
                                 const core::Tensor &target_depth_dy,
                                 const core::Tensor &target_intensity_dx,
                                 const core::Tensor &target_intensity_dy,
                                 const core::Tensor &source_vertex_map,
                                 const core::Tensor &source_mask, 
                                 const core::Tensor &target_mask,
                                 const core::Tensor &intrinsics,
                                 const core::Tensor &init_source_to_target,
                                 core::Tensor &delta,
                                 float &inlier_residual,
                                 int &inlier_count,
                                 const float depth_outlier_trunc,
                                 const float depth_huber_delta,
                                 const float intensity_huber_delta) {
    // Only Float32 is supported as of now. TODO. Support Float64.
    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();

    core::AssertTensorDtype(source_depth, supported_dtype);
    core::AssertTensorDtype(target_depth, supported_dtype);
    core::AssertTensorDtype(source_intensity, supported_dtype);
    core::AssertTensorDtype(target_intensity, supported_dtype);
    core::AssertTensorDtype(target_depth_dx, supported_dtype);
    core::AssertTensorDtype(target_depth_dy, supported_dtype);
    core::AssertTensorDtype(target_intensity_dx, supported_dtype);
    core::AssertTensorDtype(target_intensity_dy, supported_dtype);
    
    core::AssertTensorDtype(source_mask, core::Dtype::UInt8);
    core::AssertTensorDtype(target_mask, core::Dtype::UInt8);

    core:: Tensor source_mask_b = source_mask.To(core::Dtype::Bool);
    core:: Tensor target_mask_b = target_mask.To(core::Dtype::Bool);

    core::AssertTensorDevice(source_depth, device);
    core::AssertTensorDevice(target_depth, device);
    core::AssertTensorDevice(source_intensity, device);
    core::AssertTensorDevice(target_intensity, device);
    core::AssertTensorDevice(target_depth_dx, device);
    core::AssertTensorDevice(target_depth_dy, device);
    core::AssertTensorDevice(target_intensity_dx, device);
    core::AssertTensorDevice(target_intensity_dy, device);

    core::AssertTensorDevice(source_mask, device);
    core::AssertTensorDevice(target_mask, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.IsCPU()) {
        ComputeDMaskOdometryResultHybridCPU(
                source_depth, target_depth, source_intensity, target_intensity,
                target_depth_dx, target_depth_dy, target_intensity_dx,
                target_intensity_dy, source_vertex_map, source_mask_b, target_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta, intensity_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_depth.GetDevice());
        CUDA_CALL(ComputeDMaskOdometryResultHybridCUDA, source_depth, target_depth,
                  source_intensity, target_intensity, target_depth_dx,
                  target_depth_dy, target_intensity_dx, target_intensity_dy,
                  source_vertex_map, source_mask_b, target_mask_b, intrinsics_d, trans_d, delta,
                  inlier_residual, inlier_count, depth_outlier_trunc,
                  depth_huber_delta, intensity_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }
}

void ComputeSMaskOdometryResultHybrid(const core::Tensor &source_depth,
                                 const core::Tensor &target_depth,
                                 const core::Tensor &source_intensity,
                                 const core::Tensor &target_intensity,
                                 const core::Tensor &target_depth_dx,
                                 const core::Tensor &target_depth_dy,
                                 const core::Tensor &target_intensity_dx,
                                 const core::Tensor &target_intensity_dy,
                                 const core::Tensor &source_vertex_map,
                                 const core::Tensor &source_mask, 
                                 const core::Tensor &intrinsics,
                                 const core::Tensor &init_source_to_target,
                                 core::Tensor &delta,
                                 float &inlier_residual,
                                 int &inlier_count,
                                 const float depth_outlier_trunc,
                                 const float depth_huber_delta,
                                 const float intensity_huber_delta) {
    // Only Float32 is supported as of now. TODO. Support Float64.
    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();

    core::AssertTensorDtype(source_depth, supported_dtype);
    core::AssertTensorDtype(target_depth, supported_dtype);
    core::AssertTensorDtype(source_intensity, supported_dtype);
    core::AssertTensorDtype(target_intensity, supported_dtype);
    core::AssertTensorDtype(target_depth_dx, supported_dtype);
    core::AssertTensorDtype(target_depth_dy, supported_dtype);
    core::AssertTensorDtype(target_intensity_dx, supported_dtype);
    core::AssertTensorDtype(target_intensity_dy, supported_dtype);
    
    core::AssertTensorDtype(source_mask, core::Dtype::UInt8);
    core::Tensor source_mask_b = source_mask.To(core::Dtype::Bool);

    core::AssertTensorDevice(source_depth, device);
    core::AssertTensorDevice(target_depth, device);
    core::AssertTensorDevice(source_intensity, device);
    core::AssertTensorDevice(target_intensity, device);
    core::AssertTensorDevice(target_depth_dx, device);
    core::AssertTensorDevice(target_depth_dy, device);
    core::AssertTensorDevice(target_intensity_dx, device);
    core::AssertTensorDevice(target_intensity_dy, device);
    core::AssertTensorDevice(source_mask, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.IsCPU()) {
        ComputeSMaskOdometryResultHybridCPU(
                source_depth, target_depth, source_intensity, target_intensity,
                target_depth_dx, target_depth_dy, target_intensity_dx,
                target_intensity_dy, source_vertex_map, source_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta, intensity_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_depth.GetDevice());
        CUDA_CALL(ComputeSMaskOdometryResultHybridCUDA, source_depth, target_depth,
                  source_intensity, target_intensity, target_depth_dx,
                  target_depth_dy, target_intensity_dx, target_intensity_dy,
                  source_vertex_map, source_mask_b, intrinsics_d, trans_d, delta,
                  inlier_residual, inlier_count, depth_outlier_trunc,
                  depth_huber_delta, intensity_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }
}

void ComputeTMaskOdometryResultHybrid(const core::Tensor &source_depth,
                                 const core::Tensor &target_depth,
                                 const core::Tensor &source_intensity,
                                 const core::Tensor &target_intensity,
                                 const core::Tensor &target_depth_dx,
                                 const core::Tensor &target_depth_dy,
                                 const core::Tensor &target_intensity_dx,
                                 const core::Tensor &target_intensity_dy,
                                 const core::Tensor &source_vertex_map,
                                 const core::Tensor &target_mask,
                                 const core::Tensor &intrinsics,
                                 const core::Tensor &init_source_to_target,
                                 core::Tensor &delta,
                                 float &inlier_residual,
                                 int &inlier_count,
                                 const float depth_outlier_trunc,
                                 const float depth_huber_delta,
                                 const float intensity_huber_delta) {
    // Only Float32 is supported as of now. TODO. Support Float64.
    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();

    core::AssertTensorDtype(source_depth, supported_dtype);
    core::AssertTensorDtype(target_depth, supported_dtype);
    core::AssertTensorDtype(source_intensity, supported_dtype);
    core::AssertTensorDtype(target_intensity, supported_dtype);
    core::AssertTensorDtype(target_depth_dx, supported_dtype);
    core::AssertTensorDtype(target_depth_dy, supported_dtype);
    core::AssertTensorDtype(target_intensity_dx, supported_dtype);
    core::AssertTensorDtype(target_intensity_dy, supported_dtype);
    
    core::AssertTensorDtype(target_mask, core::Dtype::UInt8);
    core::Tensor target_mask_b = target_mask.To(core::Dtype::Bool);

    core::AssertTensorDevice(source_depth, device);
    core::AssertTensorDevice(target_depth, device);
    core::AssertTensorDevice(source_intensity, device);
    core::AssertTensorDevice(target_intensity, device);
    core::AssertTensorDevice(target_depth_dx, device);
    core::AssertTensorDevice(target_depth_dy, device);
    core::AssertTensorDevice(target_intensity_dx, device);
    core::AssertTensorDevice(target_intensity_dy, device);

    core::AssertTensorDevice(target_mask, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.IsCPU()) {
        ComputeTMaskOdometryResultHybridCPU(
                source_depth, target_depth, source_intensity, target_intensity,
                target_depth_dx, target_depth_dy, target_intensity_dx,
                target_intensity_dy, source_vertex_map, target_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta, intensity_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_depth.GetDevice());
        CUDA_CALL(ComputeTMaskOdometryResultHybridCUDA, source_depth, target_depth,
                  source_intensity, target_intensity, target_depth_dx,
                  target_depth_dy, target_intensity_dx, target_intensity_dy,
                  source_vertex_map, target_mask_b, intrinsics_d, trans_d, delta,
                  inlier_residual, inlier_count, depth_outlier_trunc,
                  depth_huber_delta, intensity_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }
}

void ComputeDMaskOdometryResultPointToPlane(
        const core::Tensor &source_vertex_map,
        const core::Tensor &target_vertex_map,
        const core::Tensor &target_normal_map,
        const core::Tensor &source_mask, 
        const core::Tensor &target_mask,
        const core::Tensor &intrinsics,
        const core::Tensor &init_source_to_target,
        core::Tensor &delta,
        float &inlier_residual,
        int &inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta){

    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();
    core::AssertTensorDtype(target_vertex_map, supported_dtype);
    core::AssertTensorDtype(target_normal_map, supported_dtype);
    
    core::AssertTensorDtype(source_mask, core::Dtype::UInt8);
    core::AssertTensorDtype(target_mask, core::Dtype::UInt8);
    core::Tensor source_mask_b = source_mask.To(core::Dtype::Bool);
    core::Tensor target_mask_b = target_mask.To(core::Dtype::Bool);

    core::AssertTensorDevice(target_vertex_map, device);
    core::AssertTensorDevice(target_normal_map, device);
    core::AssertTensorDevice(source_mask, device);
    core::AssertTensorDevice(target_mask, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.IsCPU()) {
        ComputeDMaskOdometryResultPointToPlaneCPU(
                source_vertex_map, target_vertex_map, target_normal_map,
                source_mask_b, target_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(device);
        CUDA_CALL(ComputeDMaskOdometryResultPointToPlaneCUDA, 
                source_vertex_map, target_vertex_map, target_normal_map,
                source_mask_b, target_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }

}

void ComputeSMaskOdometryResultPointToPlane(
        const core::Tensor &source_vertex_map,
        const core::Tensor &target_vertex_map,
        const core::Tensor &target_normal_map,
        const core::Tensor &source_mask, 
        const core::Tensor &intrinsics,
        const core::Tensor &init_source_to_target,
        core::Tensor &delta,
        float &inlier_residual,
        int &inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta){

    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();
    core::AssertTensorDtype(target_vertex_map, supported_dtype);
    core::AssertTensorDtype(target_normal_map, supported_dtype);
    
    core::AssertTensorDtype(source_mask, core::Dtype::UInt8);
    core::Tensor source_mask_b = source_mask.To(core::Dtype::Bool);

    core::AssertTensorDevice(target_vertex_map, device);
    core::AssertTensorDevice(target_normal_map, device);
    core::AssertTensorDevice(source_mask, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.IsCPU()) {
        ComputeSMaskOdometryResultPointToPlaneCPU(
                source_vertex_map, target_vertex_map, target_normal_map,
                source_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(device);
        CUDA_CALL(ComputeSMaskOdometryResultPointToPlaneCUDA, 
                source_vertex_map, target_vertex_map, target_normal_map,
                source_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }

}

void ComputeTMaskOdometryResultPointToPlane(
        const core::Tensor &source_vertex_map,
        const core::Tensor &target_vertex_map,
        const core::Tensor &target_normal_map,
        const core::Tensor &target_mask,
        const core::Tensor &intrinsics,
        const core::Tensor &init_source_to_target,
        core::Tensor &delta,
        float &inlier_residual,
        int &inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta){

    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();
    core::AssertTensorDtype(target_vertex_map, supported_dtype);
    core::AssertTensorDtype(target_normal_map, supported_dtype);
    
    core::AssertTensorDtype(target_mask, core::Dtype::UInt8);
    core::Tensor target_mask_b = target_mask.To(core::Dtype::Bool);

    core::AssertTensorDevice(target_vertex_map, device);
    core::AssertTensorDevice(target_normal_map, device);
    core::AssertTensorDevice(target_mask, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.IsCPU()) {
        ComputeTMaskOdometryResultPointToPlaneCPU(
                source_vertex_map, target_vertex_map, target_normal_map,
                target_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(device);
        CUDA_CALL(ComputeTMaskOdometryResultPointToPlaneCUDA, 
                source_vertex_map, target_vertex_map, target_normal_map,
                target_mask_b, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }

}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
