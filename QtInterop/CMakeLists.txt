cmake_minimum_required( VERSION 3.1 )
project( cgt_qt_interop )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# Add include to the set of compiler include paths
include_directories( ../core/src )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_library( cgt_qt_interop SHARED ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_qt_interop Qt5::Core Qt5::Widgets )

install( TARGETS cgt_qt_interop DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/qt_interop
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
