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

target_include_directories( cgt_core
    PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>  # <prefix>/include
	# When building libcgt itself, need to specify .. so we can pull in
	# libraries under ../third_party as #include <third_party/...>.
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

install( TARGETS cgt_core
	EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

install( DIRECTORY src/ DESTINATION include/libcgt/core
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
