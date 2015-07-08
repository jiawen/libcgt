cmake_minimum_required( VERSION 3.1 )
project( libcgt )

# On Windows, ensure that the appropriate dependencies are listed in
# CMAKE_PREFIX_PATH.

# TODO: figure out how to make a CMake library project that also creates an
# include directory and copies files appropriately. file(COPY include/foo.h
# DESTINATION "dst") seems fine.
#
# TODO: once .h files are copied out, change all includes to full pathnames.
# http://stackoverflow.com/questions/697560/how-to-copy-directory-from-source-tree-to-binary-tree
#
# How does this interact with INSTALL?
#

# TODO: on Windows, figure how to make an omnibus static library that includes
# other static libraries. I.e., Librarian / General / Additional Dependencies.

# Make sure that all subproject put all their libraries into the "lib" and
# "bin". "bin" is superfluous as libcgt is a static library.
set( OUTPUT_INCLUDE_DIRECTORY {$CMAKE_BINARY_DIR}/include )

set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin )

set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin )

# Compiler options by platform.
if( APPLE )
    # Clang only
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.7 -std=c++11 -stdlib=libc++" )
elseif( UNIX )
    if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
		# Clang
    	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++" )
	else( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
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
    option( BUILD_KINECT1 "Build Kinect v1 module." ON )
    if( BUILD_KINECT1 )
        add_subdirectory( Kinect1 )
    endif()

#    add_subdirectory( QDirectX/D3D11 )
endif()


#add_subdirectory( math )
#add_subdirectory( video )

#install( TARGETS cgt_core cgt_gl )
