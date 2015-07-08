cmake_minimum_required( VERSION 3.1 )
project( cgt_cuda )

# CUDA
find_package( CUDA REQUIRED )
# Apparently, leaving this on interferes with parallel builds.
set( CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF )

# Add include to the set of compiler include paths
include_directories( ../core/include )
include_directories( include )

if( DEFINED ENV{NVCUDASAMPLES_ROOT} )
    include_directories( $ENV{NVCUDASAMPLES_ROOT}/common/inc )
else()
    message( "Environment variable NVCUDASAMPLES_ROOT must be set." )
    return()
endif()

file( GLOB_RECURSE HEADERS include/*.h include/*.inl include/*.cuh )
file( GLOB_RECURSE SOURCES_CPP src/*.cpp )
file( GLOB_RECURSE SOURCES_CU src/*.cu )

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52 )

set( CUDA_NVCC_FLAGS_DEBUG -g;-G;-DTHRUST_DEBUG )
set( CUDA_NVCC_FLAGS_RELEASE --use_fast_math )

cuda_add_library( cgt_cuda ${HEADERS} ${SOURCES_CPP} ${SOURCES_CU} )
#file( COPY ${HEADERS} DESTINATION ${OUTPUT_INCLUDE_DIRECTORY} )
