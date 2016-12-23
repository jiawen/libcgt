cmake_minimum_required( VERSION 3.4 )
project( cgt_core )

file( GLOB_RECURSE HEADERS
	src/*.h src/*.inl
	../third_party/lodepng/lodepng.h
	../third_party/pystring/pystring.h
)
file( GLOB_RECURSE SOURCES
	src/*.cpp
	../third_party/lodepng/lodepng.cpp
	../third_party/pystring/pystring.cpp
)

add_library( cgt_core SHARED ${HEADERS} ${SOURCES} )

# TODO: add a libcgt to the include path
# http://stackoverflow.com/questions/25676277/cmake-target-include-directories-prints-an-error-when-i-try-to-add-the-source
# and
# https://cmake.org/cmake/help/v3.1/command/target_include_directories.html

# $<BUILD_INTERFACE... is for when you depend on it in build-tree mode.
# $<INSTALL_INTERFACE... is for when you depend on it in install-tree mode.

# Look here for how to declare third party libraries:
# https://github.com/google/glog/blob/master/CMakeLists.txt#L400

# http://stackoverflow.com/questions/31537602/how-to-use-cmake-to-find-and-link-to-a-library-using-install-export-and-find-pac

target_include_directories(cgt_core
    PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>  # <prefix>/include
)

install( TARGETS cgt_core
	EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

install( DIRECTORY src/ DESTINATION include/core
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)