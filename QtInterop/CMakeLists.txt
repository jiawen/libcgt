cmake_minimum_required( VERSION 3.1 )
project( cgt_qt_interop )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# Add include to the set of compiler include paths
include_directories( include )
include_directories( ../core/include )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl external/*.h )
file( GLOB_RECURSE SOURCES src/*.cpp external/*.cpp )

add_library( cgt_qt_interop ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_qt_interop cgt_core Qt5::Core Qt5::Widgets )
