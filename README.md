# onnxruntime_segment
C++
vs2019 or linux 

download onnxruntime from the NuGet  
opencv4 

https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#tensorrt
windows
.\build.bat   --cuda_version 10.2   --cudnn_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"  --cuda_home  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"  --use_tensorrt --tensorrt_home "D:\software\TensorRT-7.0.0.11_cuda10.2" 



git   clone  https://github.com/microsoft/onnxruntime.git

windows下如果想用2019
根据自己需要更改
.\onnxruntime\tools\ci_build\build.py  配置

  parser.add_argument(
        "--cmake_generator",
        choices=['Visual Studio 15 2017', 'Visual Studio 16 2019', 'Ninja'],
        default='Visual Studio 16 2019',
        help="Specify the generator that CMake invokes. "
        "This is only supported on Windows")

