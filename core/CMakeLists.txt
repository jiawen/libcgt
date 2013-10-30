cmake_minimum_required( VERSION 2.8.12 )
project( libcgt_core )

if( APPLE )
    # Clang only
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.7 -std=c++11 -stdlib=libc++" )
elseif( UNIX )
    if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
		# Clang
    	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++" )
	else( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
		# gcc, for version >= 4.7 can actually use -std=c++11
    	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
	endif()
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
elseif( WIN32 )
    # HACK: %WindowsSDKDir% environment variable?
    set( CMAKE_PREFIX_PATH "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64" )
endif()

# Qt 5
set( CMAKE_AUTOMOC ON )
find_package( Qt5Core )
find_package( Qt5Widgets )

# Add include to the set of compiler include paths
include_directories( include )

file( GLOB_RECURSE HEADERS include/*.h include/*.inl external/*.h )
file( GLOB_RECURSE SOURCES src/*.cpp external/*.cpp )

if( WIN32 )
    # Debug mode
    set( CMAKE_DEBUG_POSTFIX d )

    # Release mode
    set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_SECURE_SCL=0" )
    add_definitions( -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DUNICODE -D_UNICODE )

    option( WIN32_USE_MP "Set to ON to use multithreaded builds." ON )
    if( WIN32_USE_MP )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
    endif( WIN32_USE_MP )
endif()

set( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )

if( WIN32 )
    add_library( libcgt_core ${HEADERS} ${SOURCES} )
    qt5_use_modules( libcgt_core Core Widgets )
else( WIN32 )
    add_library( cgt_core ${HEADERS} ${SOURCES} )
    qt5_use_modules( cgt_core Core Widgets )
endif()
