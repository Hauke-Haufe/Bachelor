#include "open3d/t/geometry/RGBDMImage.h"

namespace open3d {
namespace t {
namespace geometry {

RGBDMImage &RGBDMImage::Clear() {
    color_.Clear();
    depth_.Clear();
    mask_.Clear();
    return *this;
}

bool RGBDMImage::IsEmpty() const { return color_.IsEmpty() && depth_.IsEmpty() && mask_.IsEmpty(); }

std::string RGBDMImage::ToString() const {
    return  fmt::format(
            "RGBD Image pair [{}Aligned]\n"
            "Color [size=({},{}), channels={}, format={}, device={}]\n"
            "Depth [size=({},{}), channels={}, format={}, device={}]\n",
            "Mask [size=({},{}), channels={}, format={}, device={}]",
            AreAligned() ? "" : "Not ", color_.GetCols(), color_.GetRows(),
            color_.GetChannels(), color_.GetDtype().ToString(),
            color_.GetDevice().ToString(), depth_.GetCols(), depth_.GetRows(),
            depth_.GetChannels(), depth_.GetDtype().ToString(),
            depth_.GetDevice().ToString(), 
            mask_.GetCols(), mask_.GetRows(), mask_.GetChannels(), mask_.GetDtype().ToString(),
            mask_.GetDevice().ToString());
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d