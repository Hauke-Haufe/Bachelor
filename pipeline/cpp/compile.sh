#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf build
mkdir build

#OPEN3D_DIR="/home/nb-messen-07/open3d_install/lib/cmake/Open3D"
OPEN3D_DIR="$SCRIPT_DIR/Open3D_modified/install/lib/cmake/Open3D"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=Debug .. -DOpen3D_DIR=${OPEN3D_DIR} 
make -j$(nproc)