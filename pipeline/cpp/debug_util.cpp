#include "debug_util.h"

std::string DtypeToString(const open3d::core::Dtype& dtype){
    using D = open3d::core::Dtype;

    if (dtype == D::UInt8)   return "UInt8";
    if (dtype == D::UInt16)  return "UInt16";
    if (dtype == D::UInt32)  return "UInt32";
    if (dtype == D::UInt64)  return "UInt64";
    if (dtype == D::Int8)    return "Int8";
    if (dtype == D::Int16)   return "Int16";
    if (dtype == D::Int32)   return "Int32";
    if (dtype == D::Int64)   return "Int64";
    if (dtype == D::Float32) return "Float32";
    if (dtype == D::Float64) return "Float64";
    if (dtype == D::Bool)    return "Bool";
    if (dtype == D::Undefined) return "Undefined";

    return "Unknown";
}