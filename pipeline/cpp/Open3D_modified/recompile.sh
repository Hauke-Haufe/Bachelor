#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $SCRIPT_DIR/build 
make -j4 #$(nproc) save guard for the working pc
make install 
make python-package