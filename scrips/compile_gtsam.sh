#!/bin/bash

#ändern
#INSTALL_PATH = ~/install/gtsam
#PYTHON_VERSION = 3.10.12

#doc building funktioniert nicht 
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=~/install/gtsam -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.10.12 -DGTSAM_GENERATE_DOC_XML=1 ..
make -j6 #crash auf arbeitspc sonst
doxygen build/doc/Doxyfile.in
make install -j6