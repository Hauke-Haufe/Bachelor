#!/bin/bash
set -e

OPEN3D_DIR="/home/nb-messen-07/open3d_install/lib/cmake/Open3D"
BUILD_DIR="build"

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=Debug .. -DOpen3D_DIR=${OPEN3D_DIR} 

make -j$(nproc)