This is "libcgt", my library of computer graphics tools. It is MIT licensed.

It targets Windows 8.1+ and 10, OS X Yosemite, and Android (armeabi-v7a and
arm64-v8a). It has only been extensively tested on Windows x64.

libcgt comprises a number of modules, each of which is a static (default) or
dynamic library. We list the module dependencies below.

Dependencies
============

core:

  * None.

GL:

  * GPU drivers with OpenGL >= 4.5.
  * GLEW >= 1.13.0.

CUDA:

  * NVIDIA CUDA Toolkit 7.5.
  * NVIDIA CUDA Samples (for cutil).

Kinect1x:

  * Microsoft Kinect SDK 1.8.
  * Qt (soon to be removed).

QDirectX:

  * Qt 5.
  * Microsoft DirectX SDK (10 depends on 10, 11 depends on 11)

QtInterop:

  * Qt 5.

OpenCVInterop:

  * OpenCV 3.0 or 3.1.

Experimental modules
--------------------
These modules used to work but needs some TLC to get to build again on all
platforms.

math:

  * Intel Math Kernel Library 10.3+
  * SuiteSparse 3.6.1 (http://www.cise.ufl.edu/research/sparse/SuiteSparse/)

video:

  * ffmpeg 0.5


Building for Desktop
====================

On desktop, libcgt uses CMake 3.1+.

Windows
-------

Set the `CMAKE_PREFIX_PATH` environment variable to where dependencies live. For
example, I put all my libraries under c:\work\libraries. To find GLEW, do the
following:

`set CMAKE_PREFIX_PATH=%CMAKE_PREFIX_PATH%;c:\work\libraries\glew-1.13.0;c:\work\libraries\glew-1.13.0\lib\Release\x64`

To find OpenCV, use:  
`set OpenCV_DIR=c:\work\libraries\opencv-3.1.0\build`

By CMake convention, you should put the generated Visual Studio solution in a build directory:
`$ mkdir build`  
`$ cmake-gui ..`

OS X
----

MacPorts versions of all dependencies work great out of the box. CMake is able
to find them from /opt.

By CMake convention, you should put the generated Xcode project or Makefiles in a build directory:
`$ mkdir build`  
`$ cmake-gui ..`

Ubuntu 14.04
------------

Qt 5 from the default repository is sufficient.
`$ sudo aptitude install qt5-default`

You will need download and build more recent versions of GLEW and OpenCV. I put
these libraries under ~/work/libs. I built glew under ~/work/libs/glew-1.13.0,
which contains as subdirectories bin, include, and lib. Add the following to
your .bashrc:

`set CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:~/work/libs/glew-1.13.0`

To build and install OpenCV:

`cd <opencv_dir>`  
`mkdir build`  
`cd build`  
`cmake-gui -D CMAKE_INSTALL_PREFIX=~/work/libs/opencv-3.1.0 ..`  
`make -j32`

~/work/libs/opencv-3.1.0 should contain as subdirectories bin, include, lib, and
share.

`set OpenCV_DIR=~/work/libs/opencv-3.1.0`

Apparently, gcc-4.8 has trouble building libcgt. We use Clang 3.5+ instead:

`mkdir build`  
`cd build`  
`CC=/usr/bin/clang-3.5 CXX=/usr/bin/clang++-3.5 cmake-gui-3.4.3 -DCMAKE_INSTALL_PREFIX=/mnt/ssd840/work/libs ..`

Building for Android
====================

`cd build_android`  
`./build.sh`
