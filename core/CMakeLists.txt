cmake_minimum_required( VERSION 3.4 )
project( cgt_core )

file( GLOB_RECURSE HEADERS
	*.h *.inl
	../third_party/lodepng/lodepng.h
	../third_party/pystring/pystring.h
)
file( GLOB_RECURSE SOURCES
	*.cpp
	../third_party/lodepng/lodepng.cpp
	../third_party/pystring/pystring.cpp
)

add_library( cgt_core SHARED ${HEADERS} ${SOURCES} )

if( WIN32 )
    # So that M_PI works when using #include <cmath>.
    target_compile_definitions( cgt_core
        PUBLIC _USE_MATH_DEFINES
    )
endif()
target_include_directories( cgt_core
    PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
)

install( TARGETS cgt_core
	EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

install( DIRECTORY . DESTINATION include/libcgt/core
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
