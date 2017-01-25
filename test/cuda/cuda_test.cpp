#include "libcgt/core/common/ForND.h"
#include "libcgt/core/math/Random.h"
#include "libcgt/cuda/DeviceOpaqueArray2D.h"
#include "libcgt/cuda/DeviceArray3D.h"

// TODO: connect this to a unit test framework.
void testDeviceArray3D()
{
    Random rnd( 0 );
    Vector3i size( 5, 7, 11 );
    Array3D< int > h_array( size, 0 );
    Array3D< int > h_readback( size, 0 );
    libcgt::core::for3D( size, [&] ( const Vector3i& xyz )
    {
        h_array[ xyz ] = rnd.nextIntExclusive( 256 );
    } );

    DeviceArray3D< int > d_array( size );
    copy< int >( h_array, d_array );
    copy< int >( d_array, h_readback );
    bool copyOk = true;
    libcgt::core::for3D( size, [&] ( const Vector3i& xyz )
    {
        if( h_array[ xyz ] != h_readback[ xyz ] )
        {
            copyOk = false;
            return;
        }
    } );

    DeviceArray3D< int > d_arrayCopy( d_array );
    h_readback.fill( 0 );
    copy< int >( d_arrayCopy, h_readback );
    bool copyConstructOk = true;
    libcgt::core::for3D( size, [&] ( const Vector3i& xyz )
    {
        if( h_array[ xyz ] != h_readback[ xyz ] )
        {
            copyConstructOk = false;
            return;
        }
    } );

    DeviceArray3D< int > d_arrayAssignment = d_array;
    h_readback.fill( 0 );
    copy< int >( d_arrayAssignment, h_readback );
    bool assignmentOk = true;
    libcgt::core::for3D( size, [&] ( const Vector3i& xyz )
    {
        if( h_array[ xyz ] != h_readback[ xyz ] )
        {
            assignmentOk = false;
            return;
        }
    } );

    printf( "copyOk = %d, copyConstructOk = %d, assignmentOk = %d\n",
        copyOk, copyConstructOk, assignmentOk );
}


void testDeviceOpaqueArray()
{
    Random rnd( 0 );
    Vector2i size( 5, 7 );
    Array2D< int > h_array( size, 0 );
    Array2D< int > h_readback( size, 0 );
    libcgt::core::for2D( size, [&] ( const Vector2i& xy )
    {
        h_array[ xy ] = rnd.nextIntExclusive( 256 );
    } );

    DeviceOpaqueArray2D< int > d_array( size );
    copy< int >( h_array, d_array );
    copy< int >( d_array, h_readback );
    bool copyOk = true;
    libcgt::core::for2D( size, [&] ( const Vector2i& xy )
    {
        if( h_array[ xy ] != h_readback[ xy ] )
        {
            copyOk = false;
            return;
        }
    } );

    DeviceOpaqueArray2D< int > d_arrayCopy( d_array );
    h_readback.fill( 0 );
    copy< int >( d_arrayCopy, h_readback );
    bool copyConstructOk = true;
    libcgt::core::for2D( size, [&] ( const Vector2i& xy )
    {
        if( h_array[ xy ] != h_readback[ xy ] )
        {
            copyConstructOk = false;
            return;
        }
    } );

    DeviceOpaqueArray2D< int > d_arrayAssignment = d_array;
    h_readback.fill( 0 );
    copy< int >( d_arrayAssignment, h_readback );
    bool assignmentOk = true;
    libcgt::core::for2D( size, [&] ( const Vector2i& xy )
    {
        if( h_array[ xy ] != h_readback[ xy ] )
        {
            assignmentOk = false;
            return;
        }
    } );

    printf( "copyOk = %d, copyConstructOk = %d, assignmentOk = %d\n",
        copyOk, copyConstructOk, assignmentOk );
}
