cmake_minimum_required( VERSION 3.1 )
project( cgt_opencv_interop )

find_package( OpenCV REQUIRED )

# Add include to the set of compiler include paths
include_directories( include )
include_directories( ../core/include )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl external/*.h )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_library( cgt_opencv_interop ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_opencv_interop cgt_core ${OpenCV_LIBS} )
