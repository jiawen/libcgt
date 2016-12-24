cmake_minimum_required( VERSION 3.4 )
project( cgt_cuda )

# CUDA
find_package( CUDA REQUIRED )
set( CUDA_DEPENDENCY
    "find_dependency( CUDA )" PARENT_SCOPE
)

set( CUDA_SEPARABLE_COMPILATION ON )

# Define CUDA_SAMPLES_ROOT based on platform.
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

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52 )
set( CUDA_NVCC_FLAGS_DEBUG -g;-G;-DTHRUST_DEBUG )
set( CUDA_NVCC_FLAGS_RELEASE --use_fast_math )
if( UNIX )
	set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -fPIC )
endif()

file( GLOB_RECURSE HEADERS src/*.h src/*.inl src/*.cuh )
file( GLOB_RECURSE SOURCES_CPP src/*.cpp )
file( GLOB_RECURSE SOURCES_CU src/*.cu )

# BUG(CMake): target_include_directories doesn't work with CUDA.
# https://codeyarns.com/2015/03/26/target_include_directories-does-not-work-with-cuda/
# https://gitlab.kitware.com/cmake/cmake/issues/14201
# https://cmake.org/pipermail/cmake-developers/2016-August/029298.html
include_directories( ../core/src )
include_directories( ${CUDA_SAMPLES_ROOT}/common/inc )
include_directories( ${CUDA_SAMPLES_ROOT}/src )

cuda_add_library( cgt_cuda SHARED ${HEADERS} ${SOURCES_CPP} ${SOURCES_CU} )
target_include_directories( cgt_cuda
    PUBLIC
		${CUDA_TOOLKIT_ROOT_DIR}/include
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
		$<BUILD_INTERFACE:${CUDA_SAMPLES_ROOT}/common/inc>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries( cgt_cuda
    cgt_core cgt_gl
)

install( TARGETS cgt_cuda
    EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

install( DIRECTORY src/ DESTINATION include/libcgt/cuda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" PATTERN "*.cuh"
)
