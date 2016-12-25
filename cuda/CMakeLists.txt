cmake_minimum_required( VERSION 3.7 )
project( cgt_cuda )

# BUG
message( STATUS
    "If you get an error about GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM being\
 undefined, this is due to the fact that CUDA samples includes an older\
 version of GLEW, which lacks the symbol. As a workaround, replace glew.h and\
 wglew.h with a newer version until we stop depending on samples."
)

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
		# Workaround a CMake bug where backslashes are not parsed properly
		# and we get a scary message.
        string( REPLACE "\\" "/" CUDA_SAMPLES_ROOT ${CUDA_SAMPLES_ROOT} )
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

file( GLOB_RECURSE HEADERS *.h *.inl *.cuh )
file( GLOB_RECURSE SOURCES_CPP *.cpp )
file( GLOB_RECURSE SOURCES_CU *.cu )

cuda_add_library( cgt_cuda SHARED ${HEADERS} ${SOURCES_CPP} ${SOURCES_CU} )
message( STATUS
    "You can safely ignore the warning about policy CMP0007. CMake decided\
 to put two semicolons in a row for some reason when appending to a list."
)
target_include_directories( cgt_cuda
    PUBLIC
		${CUDA_TOOLKIT_ROOT_DIR}/include
		${CUDA_SAMPLES_ROOT}/common/inc
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
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

install( DIRECTORY . DESTINATION include/libcgt/cuda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" PATTERN "*.cuh"
)