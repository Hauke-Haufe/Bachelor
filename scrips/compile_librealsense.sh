#!/bin/bash



mkdir build
cd build
cmake .. -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/home/hauke/code/Beachlor/.venv/bin/python
make -j6 
sudo make install



