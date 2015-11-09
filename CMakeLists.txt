cmake_minimum_required( VERSION 3.1 )
project( libcgt )

# Dependencies
#
# We use FindGLEW.cmake. It's amazingly good at finding glew headers/libs/dlls!
# It uses find_path and find_library. On Windows, I had:
# C:\Users\jiawen\work\libs\{bin/include/lib}, with the .dll, .h, and .lib files
# in those directories, respectively.
# I had *only* PATH=C:\Users\jiawen\work\libs\bin (as a literal). That was all
# it took to find it.

# On Windows, ensure that the appropriate dependencies are listed in
# CMAKE_PREFIX_PATH:
#
# For OpenCV, this means adding the "build" directory that comes from the OpenCV
# installer to CMAKE_PREFIX_PATH. This will set a a variable called "OpenCV_DIR"
# to said "build" directory, along with a bunch of other variables.
#
# For Kinect SDK 1.x, this means having the environment variable
# "KINECTSDK10_DIR" set. This should be set automatically by the SDK installer.

# TODO: on Windows, figure how to make an omnibus static library that includes
# other static libraries. I.e., Librarian / General / Additional Dependencies.

# Compiler options by platform.
if( APPLE )
    # Clang only
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.7 -std=c++11 -stdlib=libc++" )
elseif( UNIX )
    if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
		# Use libc++ with Clang.
    	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++" )
	else()
    	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
	endif()
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
elseif( WIN32 )
    # In debug mode, append "d" to the library name.
    set( CMAKE_DEBUG_POSTFIX d )

    # In release mode, don't use SECURE_SCL.
    set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_SECURE_SCL=0" )

    # Disable warnings and use Unicode.
    add_definitions( -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DUNICODE -D_UNICODE )

    # Add an option to use multi-threaded builds.
    option( WIN32_USE_MP "Set to ON to use multithreaded builds." ON )
    if( WIN32_USE_MP )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
    endif()
endif()

# Set INSTALL directory by platform.
if( WIN32 )
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        # CACHE means set cache variable instead of internal.
        # PATH means use a directory chooser widget.
        # The next argument is a docstring.
        # FORCE means set it even if it's already set.
        set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation Directory" FORCE)
    endif()
endif()

# TODO: can get fancy with the options: default to whether dependencies exist.
add_subdirectory( core )

option( BUILD_GL "Build GL module." ON )
if( BUILD_GL )
    add_subdirectory( GL )
endif()

option( BUILD_CUDA "Build CUDA module." ON )
if( BUILD_CUDA )
    add_subdirectory( CUDA )
endif()

option( BUILD_OPENCV_INTEROP "Build OpenCV Interop module." ON )
if( BUILD_OPENCV_INTEROP )
    add_subdirectory( OpenCVInterop )
endif()

option( BUILD_QT_INTEROP "Build Qt Interop module." ON )
if( BUILD_OPENCV_INTEROP )
    add_subdirectory( QtInterop )
endif()

if( WIN32 )
    option( BUILD_KINECT1X "Build Kinect v1.x module." ON )
    if( BUILD_KINECT1X )
        add_subdirectory( Kinect1x )
    endif()

# TODO: restore D3D11 support
#    add_subdirectory( QDirectX/D3D11 )
endif()


# TODO: restore MKL support or replace with Eigen
#add_subdirectory( math )

# TODO: restore video support
#add_subdirectory( video )
