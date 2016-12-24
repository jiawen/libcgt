cmake_minimum_required( VERSION 3.1 )

# TODO: Kinect Developer Toolkit.

if( WIN32 )
    find_path( Kinect1x_ROOT inc/NuiApi.h
        PATHS ENV KINECTSDK10_DIR
        DOC "Kinect 1.x SDK directory" )

    # Find headers.
    find_path( Kinect1x_INCLUDE_DIR NuiApi.h
        PATHS ${Kinect1x_ROOT}/inc
        DOC "Kinect 1.x SDK include dir" )

    # Find lib based on architecture.
    if( CMAKE_CL_64 )
        set( LIB_SUFFIX amd64 )
    else()
        set( LIB_SUFFIX x86 )
    endif()

    find_path( Kinect1x_LIB_DIR Kinect10.lib
        PATHS ${Kinect1x_ROOT}/lib
        PATH_SUFFIXES ${LIB_SUFFIX}
        DOC "Kinect 1.x SDK lib dir" )

    # Find assemblies / dll directory for .NET.
    find_path( Kinect1x_ASSEMBLIES_DIR Microsoft.Kinect.dll
        PATHS ${Kinect1x_ROOT}/Assemblies
        DOC "Kinect 1.x SDK assemblies dir" )

    # Tell CMake to find the "Kinect10.lib" library.
    find_library( Kinect1x_LIBRARY
        NAMES Kinect10
        PATHS ${Kinect1x_LIB_DIR}
        NO_DEFAULT_PATH )

    # Now that we have the dirs and libs, set them for export.
    set( Kinect1x_INCLUDE_DIRS ${Kinect1x_INCLUDE_DIR} )
    set( Kinect1x_LIBRARIES ${Kinect1x_LIBRARY} )
    include( FindPackageHandleStandardArgs )
    find_package_handle_standard_args( KINECT1X
        REQUIRED_VARS Kinect1x_INCLUDE_DIR Kinect1x_LIBRARIES )

    if( Kinect1x_FOUND AND NOT TARGET Kinect1x::Kinect1x )
        add_library( Kinect1x::Kinect1x UNKNOWN IMPORTED )
        set_target_properties( Kinect1x::Kinect1x PROPERTIES
            IMPORTED_LOCATION "${Kinect1x_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${Kinect1x_INCLUDE_DIRS}" )
    endif()
    mark_as_advanced( ${Kinect1x_INCLUDE_DIR} ${Kinect1x_LIBRARY} )

else()
    message( FATAL_ERROR "Kinect SDK is only supported on windows." )
endif()
