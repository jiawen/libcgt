cmake_minimum_required( VERSION 3.1 )
project( cgt_gl )

# GLEW
find_package( GLEW REQUIRED )
include_directories( ${GLEW_INCLUDE_DIRS} )

include_directories( ../core/src )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_library( cgt_gl ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_gl cgt_core GLEW::GLEW )

install( TARGETS cgt_gl DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/gl
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
