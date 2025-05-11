#!/bin/bash
set -e
rm -rf build
mkdir build

#OPEN3D_DIR="/home/nb-messen-07/open3d_install/lib/cmake/Open3D"
OPEN3D_DIR="/home/hauke-haufe/src/Open3D/install/lib/cmake/Open3D"
BUILD_DIR="build"

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=Debug .. -DOpen3D_DIR=${OPEN3D_DIR} 

make -j$(nproc)