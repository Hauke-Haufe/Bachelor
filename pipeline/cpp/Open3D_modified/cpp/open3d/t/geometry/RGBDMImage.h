#pragma once

#include "open3d/geometry/RGBDImage.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace t {
namespace geometry {

/// \brief RGBDMImage A triple of color, depth and a mask
///
/// For most processing, the image triple should be aligned (same viewpoint and
/// resolution).
class RGBDMImage : public Geometry {
public:
    /// \brief Default Comnstructor.
    RGBDMImage() : Geometry(Geometry::GeometryType::RGBDMImage, 2) {}

    /// \brief Parameterized Constructor.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    /// \param mask The mask image.
    /// \param aligned Are the two images aligned (same viewpoint and
    /// resolution)?
    RGBDMImage(const Image &color, const Image &depth, const Image &mask, bool aligned = true)
        : Geometry(Geometry::GeometryType::RGBDMImage, 2),
          color_(color),
          depth_(depth),
          mask_(mask),
          aligned_(aligned) {
        if (color.GetRows() != depth.GetRows() ||
            color.GetRows() != mask.GetRows() ||
            color.GetCols() != mask.GetCols() ||
            color.GetCols() != depth.GetCols()) {
            aligned_ = false;
            utility::LogWarning(
                    "Aligned image triple must have the same resolution.");
        }
        if (!(mask.GetDtype() == core::Dtype::UInt8 || mask.GetDtype() == core::Dtype::Bool)){
            utility::LogError(
                "The Dtype of the mask must be UInt8 not {}", mask.GetDtype().ToString());
        }
    }

    core::Device GetDevice() const override {
        core::Device color_device = color_.GetDevice();
        core::Device depth_device = depth_.GetDevice();
        core::Device mask_device = mask_.GetDevice();
        if (color_device != depth_device) {
            utility::LogError(
                    "Color {} and depth {} are not on the same device.",
                    color_device.ToString(), depth_device.ToString());
        }
        else if (color_device != mask_device){
            utility::LogError(
                    "Color {} and mask {} are not on the same device.",
                    color_device.ToString(), depth_device.ToString());
        }
        return color_device;
    }

    ~RGBDMImage() override{};

    /// Clear stored data.
    RGBDMImage &Clear() override;

    /// Is any data stored?
    bool IsEmpty() const override;

    /// Are the depth, color and mask images aligned (same viewpoint and resolution)?
    bool AreAligned() const { return aligned_; }

    /// Compute min 2D coordinates for the data (always {0,0}).
    core::Tensor GetMinBound() const {
        return core::Tensor::Zeros({2}, core::Int64);
    }

    /// Compute max 2D coordinates for the data.

    //ToDO change this function
    core::Tensor GetMaxBound() const {
        return core::Tensor(
                std::vector<int64_t>{color_.GetCols() + depth_.GetCols(),
                                     color_.GetRows()},
                {2}, core::Int64);
    }

    /// Transfer the RGBDM image to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new image is always created; if false, the
    /// copy is avoided when the original image is already on the target
    /// device.
    RGBDMImage To(const core::Device &device, bool copy = false) const {
        return RGBDMImage(color_.To(device, copy), depth_.To(device, copy), mask_.To(device, copy),
                         aligned_);
    }

    /// Returns copy of the RGBDM image on the same device.
    RGBDMImage Clone() const { return To(color_.GetDevice(), /*copy=*/true); }

    /// Convert to the legacy RGBDMImage format.
    open3d::geometry::RGBDImage ToLegacy() const {
        utility::LogError("This function isnt supported for RGBDM Images. Dont use it.");
        return open3d::geometry::RGBDImage();
    }

    /// Text description.
    std::string ToString() const;

public:
    /// The color image.
    Image color_;
    /// The depth image.
    Image depth_;
    ///The mask image
    Image mask_;
    /// Are the depth and color images aligned (same viewpoint and resolution)?
    bool aligned_ = true;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d