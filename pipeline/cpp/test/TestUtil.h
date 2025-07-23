#pragma once
#include <filesystem>
#include <open3d/core/Tensor.h>
#include "open3d/t/pipelines/odometry/RGBDMOdometry.h"

int CountFilesDir(std::filesystem::path dirPath);

bool IsLargeRotation(const open3d::core::Tensor& trans, double angle_threshold_deg = 10.0);

bool IsLargeTranslation(const open3d::core::Tensor& transformation, float threshold = 0.3f);

bool AreTransformationsSimilar(const open3d::core::Tensor& transform1, const open3d::core::Tensor& transform2, float threshold = 1e-5);
