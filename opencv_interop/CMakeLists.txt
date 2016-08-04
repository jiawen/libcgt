cmake_minimum_required( VERSION 3.4 )
project( cgt_opencv_interop )

find_package( OpenCV REQUIRED )

# Add include to the set of compiler include paths
include_directories( ../core/src )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_library( cgt_opencv_interop SHARED ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_opencv_interop cgt_core ${OpenCV_LIBS} )

install( TARGETS cgt_opencv_interop DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/opencv_interop
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
