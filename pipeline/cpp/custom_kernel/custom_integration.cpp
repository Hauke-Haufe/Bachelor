#include <open3d/Open3D.h>
#include "custom_integration.h"
//#include "custom_inetgrationCUDA.cu"

using namespace open3d;

    #define DISPATCH_VALUE_DTYPE_TO_TEMPLATE(WEIGHT_DTYPE, COLOR_DTYPE, ...)    \
    [&] {                                                                   \
        if (WEIGHT_DTYPE == open3d::core::Float32 &&                        \
            COLOR_DTYPE == open3d::core::Float32) {                         \
            using weight_t = float;                                         \
            using color_t = float;                                          \
            using label_t = uint8_t;                                       \
            return __VA_ARGS__();                                           \
        } else if (WEIGHT_DTYPE == open3d::core::UInt16 &&                  \
                   COLOR_DTYPE == open3d::core::UInt16) {                   \
            using weight_t = uint16_t;                                      \
            using color_t = uint16_t;                                       \
            using label_t = uint8_t;                                     \
            return __VA_ARGS__();                                           \
        } else {                                                            \
            utility::LogError(                                              \
                    "Unsupported value data type combination. Expected "    \
                    "(float, float) or (uint16, uint16), but received ({} " \
                    "{}).",                                                 \
                    WEIGHT_DTYPE.ToString(), COLOR_DTYPE.ToString());       \
        }                                                                   \
    }()

#define DISPATCH_INPUT_DTYPE_TO_TEMPLATE(DEPTH_DTYPE, COLOR_DTYPE, ...)        \
    [&] {                                                                      \
        if (DEPTH_DTYPE == open3d::core::Float32 &&                            \
            COLOR_DTYPE == open3d::core::Float32) {                            \
            using input_depth_t = float;                                       \
            using input_color_t = float;                                       \
            using input_label_t = uint8_t;                                     \
            return __VA_ARGS__();                                              \
        } else if (DEPTH_DTYPE == open3d::core::UInt16 &&                      \
                   COLOR_DTYPE == open3d::core::UInt8) {                       \
            using input_depth_t = uint16_t;                                    \
            using input_color_t = uint8_t;                                      \
            using input_label_t = uint8_t;                                      \
            return __VA_ARGS__();                                              \
        } else {                                                               \
            utility::LogError(                                                 \
                    "Unsupported input data type combination. Expected "       \
                    "(float, float) or (uint16, uint8), but received ({} {})", \
                    DEPTH_DTYPE.ToString(), COLOR_DTYPE.ToString());           \
        }                                                                      \
    }()
                                                    
void custom_Integrate(const core::Tensor& depth,
            const core::Tensor& color,
            const core::Tensor& label,
            const core::Tensor& block_indices,
            const core::Tensor& block_keys,
            t::geometry::TensorMap& block_value_map,
            const core::Tensor& depth_intrinsic,
            const core::Tensor& color_intrinsic,
            const core::Tensor& extrinsic,
            index_t resolution,
            float voxel_size,
            float sdf_trunc,
            float depth_scale,
            float depth_max) {

    using tsdf_t = float;
    core::Dtype block_weight_dtype = core::Dtype::Float32;
    core::Dtype block_color_dtype = core::Dtype::Float32;
    core::Dtype block_label_dtype = core::Dtype::UInt8;

    if (block_value_map.Contains("weight")) {
        block_weight_dtype = block_value_map.at("weight").GetDtype();
    }
    if (block_value_map.Contains("color")) {
        block_color_dtype = block_value_map.at("color").GetDtype();
    }
    if (block_value_map.Contains("label")){
        block_label_dtype = block_value_map.at("label").GetDtype();
    }

    core::Dtype input_depth_dtype = depth.GetDtype();
    core::Dtype input_color_dtype = (input_depth_dtype == core::Dtype::Float32)
                                            ? core::Dtype::Float32
                                            : core::Dtype::UInt8;
    core::Dtype input_label_dtype = label.GetDtype();

    if (color.NumElements() > 0) {
        input_color_dtype = color.GetDtype();
    }


    if (depth.IsCPU()) {

        DISPATCH_INPUT_DTYPE_TO_TEMPLATE(
            input_depth_dtype, input_color_dtype, [&] {
                DISPATCH_VALUE_DTYPE_TO_TEMPLATE(
                        block_weight_dtype, block_color_dtype, [&] {
                            CustomIntegrateCPU<input_depth_t, input_color_t, input_label_t,
                                         tsdf_t, weight_t, color_t, label_t>(
                                    depth, color, label, block_indices, block_keys,
                                    block_value_map, depth_intrinsic,
                                    color_intrinsic, extrinsic, resolution,
                                    voxel_size, sdf_trunc, depth_scale,
                                    depth_max);
                        });
            });

    } else if (depth.IsCUDA()) {
        
        
        DISPATCH_INPUT_DTYPE_TO_TEMPLATE(
            input_depth_dtype, input_color_dtype, [&] {
                DISPATCH_VALUE_DTYPE_TO_TEMPLATE(
                        block_weight_dtype, block_color_dtype, [&] {
                            CustomIntegrateCUDA<input_depth_t, input_color_t, input_label_t,
                                         tsdf_t, weight_t, color_t, label_t>(
                                    depth, color, label, block_indices, block_keys,
                                    block_value_map, depth_intrinsic,
                                    color_intrinsic, extrinsic, resolution,
                                    voxel_size, sdf_trunc, depth_scale,
                                    depth_max);
                        });
            });
            
    } else {
        utility::LogError("Unimplemented device");
    }
};


