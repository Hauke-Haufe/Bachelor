mkdir lib/Open3d/build
cd lib/Open3d/build
cmake -DBUILD_CUDA_MODULE=true -DPython3_ROOT=C:\Users\Haufe\Desktop\beachlor\code\.venv\Scripts\python.exe  -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX="C:\Open3d" ..
cmake --build . --config Release --target python-package   
cd .. 
pip install lib/python-package
