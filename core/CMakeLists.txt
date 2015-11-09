cmake_minimum_required( VERSION 3.1 )
project( cgt_core )

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# Add include to the set of compiler include paths
include_directories( src )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl third_party/*.h )
file( GLOB_RECURSE SOURCES src/*.cpp third_party/*.cpp )

add_library( cgt_core ${HEADERS} ${SOURCES} )
target_link_libraries( cgt_core Qt5::Core Qt5::Widgets )

install( TARGETS cgt_core DESTINATION lib )
install( DIRECTORY src/ DESTINATION include/core
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
install( DIRECTORY third_party/ DESTINATION include/third_party
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl" )
