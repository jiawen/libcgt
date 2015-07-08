cmake_minimum_required( VERSION 3.1 )
project( cgt_gl )

# GLEW
find_package( GLEW REQUIRED )
include_directories( ${GLEW_INCLUDE_DIRS} )

include_directories( include )
include_directories( ../core/include )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl external/*.h )
file( GLOB_RECURSE SOURCES src/*.cpp external/*.cpp )

# TODO: not project_binary, write to build dir?
#set( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )

add_library( cgt_gl ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_gl cgt_core GLEW::GLEW )
