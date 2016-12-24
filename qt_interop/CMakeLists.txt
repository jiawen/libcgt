cmake_minimum_required( VERSION 3.4 )
project( cgt_qt_interop )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core REQUIRED )
find_package( Qt5Widgets REQUIRED )
set( Qt5Core_DEPENDENCY
    "find_dependency( Qt5Core )" PARENT_SCOPE
)
set( Qt5Widgets_DEPENDENCY
    "find_dependency( Qt5Widgets )" PARENT_SCOPE
)

# Add include to the set of compiler include paths
# include_directories( ../core/src )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_library( cgt_qt_interop SHARED ${HEADERS} ${SOURCES} )
target_include_directories( cgt_qt_interop
    PUBLIC		
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries( cgt_qt_interop
    cgt_core Qt5::Core Qt5::Widgets
)

install( TARGETS cgt_qt_interop
    EXPORT libcgt-targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)
install( DIRECTORY src/ DESTINATION include/libcgt/qt_interop
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)
