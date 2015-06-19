cmake_minimum_required( VERSION 2.8 )
project( libcgt_cuda )

# Debug mode
set( CMAKE_DEBUG_POSTFIX d )

# Release mode
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_SECURE_SCL=0" )

# CUDA
find_package( CUDA REQUIRED )

# Add include to the set of compiler include paths
include_directories( ../core/include )
include_directories( include )
include_directories( $ENV{NVCUDASAMPLES_ROOT}/common/inc )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl include/*.cuh )
file( GLOB_RECURSE SOURCES_CPP src/*.cpp )
file( GLOB_RECURSE SOURCES_CU src/*.cu )

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30 )

#if( MSVC11 )
#    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--cl-version 2010 )
#endif()

set( CUDA_NVCC_FLAGS_DEBUG -g;-G;-DTHRUST_DEBUG )
set( CUDA_NVCC_FLAGS_RELEASE --use_fast_math )

add_definitions( -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DUNICODE -D_UNICODE )

# if( WIN32 )
#     option( WIN32_USE_MP "Set to ON to use multithreaded builds." ON )
#     if( WIN32_USE_MP )
#         set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
#     endif( WIN32_USE_MP )
# endif( WIN32 )

cuda_add_library( libcgt_cuda ${HEADERS} ${SOURCES_CPP} ${SOURCES_CU} )
