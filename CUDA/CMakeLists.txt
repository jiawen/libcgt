cmake_minimum_required( VERSION 3.1 )
project( cgt_cuda )

# CUDA
find_package( CUDA REQUIRED )

set( CUDA_SEPARABLE_COMPILATION ON )

# Add include to the set of compiler include paths
include_directories( ../core/src )
include_directories( include )

# Define CUDA_SAMPLES_ROOT:
if( WIN32 )
    if( DEFINED ENV{NVCUDASAMPLES_ROOT} )
        set( CUDA_SAMPLES_ROOT $ENV{NVCUDASAMPLES_ROOT} )
    else()
        message( "Environment variable NVCUDASAMPLES_ROOT must be set." )
        return()
    endif()
    # Apparently, leaving this on interferes with parallel builds.
    set( CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF )
else()
    set( CUDA_SAMPLES_ROOT ${CUDA_TOOLKIT_ROOT_DIR}/samples )
endif()

include_directories( ${CUDA_SAMPLES_ROOT}/common/inc )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl src/*.cuh )
file( GLOB_RECURSE SOURCES_CPP src/*.cpp )
file( GLOB_RECURSE SOURCES_CU src/*.cu )

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -fPIC -gencode arch=compute_52,code=sm_52 )

set( CUDA_NVCC_FLAGS_DEBUG -g;-G;-DTHRUST_DEBUG )
set( CUDA_NVCC_FLAGS_RELEASE --use_fast_math )

cuda_add_library( cgt_cuda SHARED ${HEADERS} ${SOURCES_CPP} ${SOURCES_CU} )

install( TARGETS cgt_cuda DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/cuda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" PATTERN "*.cuh" )
