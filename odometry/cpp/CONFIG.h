#pragma once

#include <unordered_map>
#include <stdexcept>
#include <iostream>

#include "open3d/t/pipelines/odometry/RGBDMOdometry.h"

#include "TestDataset.h"

using namespace open3d;

enum class MaskMethod{
     SourceMask,
     TargetMask,
     CompleteMask, 
     NoMask
};

enum class SlamMethod{
    Raw, 
    Maskout, 
    WeightMaskout
};


inline std::unordered_map<std::string, t::pipelines::odometry::Method> method_map = {
    {"Intensity", t::pipelines::odometry::Method::Intensity},
    {"Hybrid", t::pipelines::odometry::Method::Hybrid},
    {"P2P", t::pipelines::odometry::Method::PointToPlane}
};

// Function to retrieve the method from the global map
inline t::pipelines::odometry::Method get_method(std::string choosen_method) {
    auto it = method_map.find(choosen_method);
    if (it == method_map.end()) {
        throw std::runtime_error("Not a valid method");
    }
    return it->second;
}


inline std::unordered_map<std::string, MaskMethod> maskmethod_map = {
    {"Source", MaskMethod::SourceMask},
    {"Target", MaskMethod::TargetMask},
    {"Both", MaskMethod::CompleteMask}, 
    {"NoMask", MaskMethod::NoMask }
};

inline MaskMethod get_maskmethod(std::string choosen_method) {
    auto it = maskmethod_map.find(choosen_method);
    if (it == maskmethod_map.end()) {
        throw std::runtime_error("Not a valid method");
    }
    return it->second;
};

inline std::unordered_map<std::string, SlamMethod> slammethod_map = {
    {"Raw", SlamMethod::Raw},
    {"Maskout", SlamMethod::Maskout},
    {"WeightMaskout", SlamMethod::WeightMaskout}
};

inline SlamMethod get_slammethod(std::string choosen_method) {
    auto it = slammethod_map.find(choosen_method);
    if (it == slammethod_map.end()) {
        throw std::runtime_error("Not a valid method");
    }
    return it->second;
}

inline std::unordered_map<std::string, core::Device> device_map = {
    {"CUDA", core::Device("CUDA:0")},
    {"CPU", core::Device("CPU:0")},
};

inline core::Device get_device(std::string choosen_method) {
    auto it = device_map.find(choosen_method);
    if (it == device_map.end()) {
        throw std::runtime_error("Not a valid method");
    }
    return it->second;
};

inline std::unordered_map<std::string, SubDataset> dataset_map = {
    {"static_xyz", SubDataset::static_xyz}, 
    {"walking_xyz", SubDataset::walking_xyz},
    {"walking_halfsphere", SubDataset::walking_halfsphere},
    {"walking_rpy", SubDataset::walking_rpy}, 
    {"walking_static", SubDataset::walking_static}
};

inline SubDataset get_dataset(std::string choosen_dataset) {
    auto it = dataset_map.find(choosen_dataset);
    if (it == dataset_map.end()) {
        throw std::runtime_error("Not a valid dataset");
    }
    return it->second;
};
