cmake_minimum_required( VERSION 3.4 )
project( libcgt CXX )

# Compiler options by platform.
if( APPLE )
    # Clang only
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++" )
elseif( UNIX )
    message( "CMAKE_CXX_COMPILER_ID is: ${CMAKE_CXX_COMPILER_ID}" )
    if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
        # Use libc++ with Clang.
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++" )
    else()
        # Use the default stdlib with other compilers.
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
        message( FATAL_ERROR "libcgt currently only compiles with Clang and libc++." )
    endif()
elseif( WIN32 )
    set( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true )

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

# Set CMAKE_INSTALL_PREFIX on Windows.
if( WIN32 )
    if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
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
    add_subdirectory( cuda )
endif()

option( BUILD_OPENCV_INTEROP "Build OpenCV Interop module." ON )
if( BUILD_OPENCV_INTEROP )
    add_subdirectory( opencv_interop )
endif()

option( BUILD_QT_INTEROP "Build Qt Interop module." ON )
if( BUILD_QT_INTEROP )
    add_subdirectory( qt_interop )
endif()

option( BUILD_CAMERA_WRAPPERS "Build Camera Wrappers module." ON )
if( BUILD_CAMERA_WRAPPERS )
    add_subdirectory( camera_wrappers )
endif()

# TODO: restore D3D11 support
#    add_subdirectory( QDirectX/D3D11 )

# TODO: restore MKL support or replace with Eigen
#add_subdirectory( math )

# TODO: restore video support
#add_subdirectory( video )
