cmake_minimum_required( VERSION 3.4 )
project( cgt_gl )

# OpenGL and GLEW
find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )
set( OpenGL_DEPENDENCY
    "find_dependency( OpenGL )" PARENT_SCOPE
)
set( GLEW_DEPENDENCY
    "find_dependency( GLEW )" PARENT_SCOPE
)

add_definitions( -D GL_PLATFORM_45 )

file( GLOB_RECURSE HEADERS
    src/common/*.h src/common/*.inl
    src/GL_45/*.h src/GL_45/*.inl
    src/GL_45/drawables/*.h src/GL_45/drawables/*.inl
)
file( GLOB_RECURSE SOURCES
    src/common/*.cpp
    src/GL_45/*.cpp
    src/GL_45/drawables/*.cpp
)

add_library( cgt_gl SHARED ${HEADERS} ${SOURCES} )
set_property( TARGET cgt_gl PROPERTY CXX_STANDARD 11 )
target_include_directories( cgt_gl
    PUBLIC		
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/common>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/GL_45>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries( cgt_gl
    cgt_core ${OPENGL_LIBRARIES} GLEW::GLEW
)

install( TARGETS cgt_gl
	EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)
install( DIRECTORY src/common DESTINATION include/GL
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
install( DIRECTORY src/GL_45 DESTINATION include/GL
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
