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

file( GLOB HEADERS
    *.h *.inl
    GL_45/*.h GL_45/*.inl
    GL_45/drawables/*.h GL_45/drawables/*.inl
)
file( GLOB SOURCES
    *.cpp
    GL_45/*.cpp
    GL_45/drawables/*.cpp
)

add_library( cgt_gl SHARED ${HEADERS} ${SOURCES} )
set_property( TARGET cgt_gl PROPERTY CXX_STANDARD 11 )

# Declare that We're using OpenGL 4.5 and so should all dependents.
target_compile_definitions( cgt_gl
    PUBLIC GL_PLATFORM_45
)	    
target_include_directories( cgt_gl
    PUBLIC		
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
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
install( DIRECTORY . DESTINATION include/libcgt/GL
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
install( DIRECTORY GL_45 DESTINATION include/libcgt/GL
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
