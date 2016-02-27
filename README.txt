This is "libcgt", my library of computer graphics tools, which I hope you will find useful.

It targets Windows 8.1+ and 10, OS X Yosemite, and Android (armeabi-v7a and
arm64-v8a). It has only been extensively tested on Windows x64.

Build system:

Desktop:

libcgt uses CMake 3.1.
qmake is now in legacy mode and not really tested. It will be removed soon.

On Windows, be sure to set the CMAKE_PREFIX_PATH environment variable to
set "base paths" for external libraries. Some tips:

- GLEW:
set CMAKE_PREFIX_PATH=%CMAKE_PREFIX_PATH%;c:\work\libraries\glew-1.12.0;c:\work\libraries\glew-1.12.0\lib\Release\x64

$ mkdir build
$ cmake-gui ..

Android:

Run build_android/build.sh

Dependencies:

core:
Qt >= 5.4, tested with Qt 5.4.

GL:
GPU drivers with OpenGL >= 4.5.
GLEW >= 1.12.0.

CUDA:
NVIDIA CUDA Toolkit 4.0
NVIDIA GPU Computing SDK (for cutil)

QDirectX:
Microsoft DirectX SDK (10 depends on 10, 11 depends on 11)

math:
Intel Math Kernel Library 10.3
SuiteSparse 3.6.1 (http://www.cise.ufl.edu/research/sparse/SuiteSparse/)

The following has not been touched in some time:

GL:
GLEW 1.7.0

Cg:
NVIDIA Cg 2.0

video:
ffmpeg 0.5
