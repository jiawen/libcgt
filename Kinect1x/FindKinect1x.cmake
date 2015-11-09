cmake_minimum_required( VERSION 3.1 )

# TODO: Kinect Developer Toolkit.

if( WIN32 )
    find_path( KINECT1X_ROOT inc/NuiApi.h
        PATHS ENV KINECTSDK10_DIR
        DOC "Kinect 1.x SDK directory" )

    # Find headers.
    find_path( KINECT1X_INCLUDE_DIR NuiApi.h
        PATHS ${KINECT1X_ROOT}/inc
        DOC "Kinect 1.x SDK include dir" )

    # Find lib based on architecture.
    if( CMAKE_CL_64 )
        set( LIB_SUFFIX amd64 )
    else()
        set( LIB_SUFFIX x86 )
    endif()

    find_path( KINECT1X_LIB_DIR Kinect10.lib
        PATHS ${KINECT1X_ROOT}/lib
        PATH_SUFFIXES ${LIB_SUFFIX}
        DOC "Kinect 1.x SDK lib dir" )

    # Find assemblies / dll directory for .NET.
    find_path( KINECT1X_ASSEMBLIES_DIR Microsoft.Kinect.dll
        PATHS ${KINECT1X_ROOT}/Assemblies
        DOC "Kinect 1.x SDK assemblies dir" )

    # Tell CMake to find the "Kinect10.lib" library.
    find_library( KINECT1X_LIBRARY
        NAMES Kinect10
        PATHS ${KINECT1X_LIB_DIR}
        NO_DEFAULT_PATH )

    # Now that we have the dirs and libs, set them for export.
    set( KINECT1X_INCLUDE_DIRS ${KINECT1X_INCLUDE_DIR} )
    set( KINECT1X_LIBRARIES ${KINECT1X_LIBRARY} )
    include( FindPackageHandleStandardArgs )
    find_package_handle_standard_args( KINECT1X
        REQUIRED_VARS KINECT1X_INCLUDE_DIR KINECT1X_LIBRARIES )

    if( KINECT1X_FOUND AND NOT TARGET Kinect1x::Kinect1x )
        add_library( Kinect1x::Kinect1x UNKNOWN IMPORTED )
        set_target_properties( Kinect1x::Kinect1x PROPERTIES
            IMPORTED_LOCATION "${KINECT1X_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${KINECT1X_INCLUDE_DIRS}" )
    endif()
    mark_as_advanced( ${KINECT1X_INCLUDE_DIR} ${KINECT1X_LIBRARY} )

else()
    message( FATAL_ERROR "Kinect SDK is only supported on windows." )
endif()
