#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
//#include "open3d/core/hashmap/CUDA/StdGPUHashBackend.h"
//#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/VoxelBlockGrid.h"
#include "open3d/t/geometry/kernel/VoxelBlockGridImpl.h"
#include "open3d/utility/Logging.h"


#include "custom_integrationImpl.h"
#include "custom_integration.h"


using namespace open3d;

#define FN_ARGUMENTS                                                          \
    const core::Tensor &depth, const core::Tensor &color, const core::Tensor& label,\
            const core::Tensor &indices, const core::Tensor &block_keys,      \
            t::geometry::TensorMap &value_tensor_map, const core::Tensor &depth_intrinsic, \
            const core::Tensor &color_intrinsic,                              \
            const core::Tensor &extrinsic, index_t resolution,                \
            float voxel_size, float sdf_trunc, float depth_scale,             \
            float depth_max

template void CustomIntegrateCUDA<uint16_t, uint8_t, uint8_t, float, uint16_t, uint16_t, uint8_t>(
        FN_ARGUMENTS);
template void CustomIntegrateCUDA<uint16_t, uint8_t, uint8_t, float, float, float, uint8_t>(
        FN_ARGUMENTS);
template void CustomIntegrateCUDA<float, float, uint8_t, float, uint16_t, uint16_t, uint8_t>(
        FN_ARGUMENTS);
template void CustomIntegrateCUDA<float, float, uint8_t, float, float, float, uint8_t>(FN_ARGUMENTS);

#undef FN_ARGUMENTS