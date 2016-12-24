cmake_minimum_required( VERSION 3.4 )
project( cgt_opencv_interop )

find_package( OpenCV REQUIRED )
set( OpenCV_DEPENDENCY
    "find_dependency( OpenCV )" PARENT_SCOPE
)

file( GLOB_RECURSE HEADERS *.h *.inl )
file( GLOB_RECURSE SOURCES *.cpp )

add_library( cgt_opencv_interop SHARED ${HEADERS} ${SOURCES} )
target_include_directories(cgt_core
    PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries( cgt_opencv_interop cgt_core ${OpenCV_LIBS} )

install( TARGETS cgt_opencv_interop
    EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

install( DIRECTORY . DESTINATION include/libcgt/opencv_interop
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
