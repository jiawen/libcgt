cmake_minimum_required( VERSION 3.1 )
project( cgt_gl )

# GLEW
find_package( GLEW REQUIRED )
include_directories( ${GLEW_INCLUDE_DIRS} )

add_definitions( -D GL_PLATFORM_45 )

include_directories( ../core/src )
include_directories( src/common )
include_directories( src/GL45 )

file( GLOB_RECURSE HEADERS src/common/*.h src/common/*.inl src/GL_45/*.h
    src/GL_45/*.inl )
file( GLOB_RECURSE SOURCES src/common/*.cpp src/GL_45/*.cpp )

add_library( cgt_gl ${HEADERS} ${SOURCES} )
set_property( TARGET cgt_gl PROPERTY CXX_STANDARD 11 )
target_link_libraries( cgt_gl cgt_core GLEW::GLEW )

install( TARGETS cgt_gl DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/gl
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )