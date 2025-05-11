#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#deletes build cache
rm -rf $SCRIPT_DIR/build
mkdir $SCRIPT_DIR/build

#created install dir 
mkdir -p $SCRIPT_DIR/install 

#compiles Open3d 
cd $SCRIPT_DIR/build 
cmake -DBUILD_CUDA_MODULE=true -DCMAKE_INSTALL_PREFIX=$SCRIPT_DIR/install -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install 