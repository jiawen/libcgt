#include "KinectUtils.h"

#include <common/ArrayUtils.h>
#include <geometry/Plane3f.h>

namespace libcgt { namespace camera_wrappers { namespace kinect1x {

#if 0
// TODO(jiawen): EuclideanTransform
Matrix4f kinectToWorld( const NUI_SKELETON_FRAME& frame )
{
    return worldToKinect( frame ).inverse();
}

Matrix4f worldToKinect( const NUI_SKELETON_FRAME& frame )
{
    Plane3f plane( frame.vFloorClipPlane.x, frame.vFloorClipPlane.y,
        frame.vFloorClipPlane.z, frame.vFloorClipPlane.w );

    // The x axis is the same.
    Vector3f xKinect( 1, 0, 0 );
    // In the Kinect's frame, y is not (0,0,1).
    Vector3f yKinect( frame.vFloorClipPlane.x, frame.vFloorClipPlane.y,
        frame.vFloorClipPlane.z );
    Vector3f zKinect = Vector3f::cross( xKinect, yKinect );

    // build a world to Kinect matrix
    Matrix4f worldToKinect;
    worldToKinect.setCol( 0, Vector4f( xKinect, 0 ) );
    // y axis is yKinect
    worldToKinect.setCol( 1, Vector4f( yKinect, 0 ) );
    // z axis is cross product
    worldToKinect.setCol( 2, Vector4f( zKinect, 0 ) );

    // In the world, the Kinect is at the point on the plane closest to the
    // origin.
    worldToKinect.setCol( 3, Vector4f( ( plane.pointOnPlane() ), 1 ) );

    return worldToKinect;
}
#endif

void rawDepthMapToMeters( Array2DReadView< uint16_t > rawDepth,
    Array2DWriteView< float > outputMeters, bool flipX, bool flipY,
    int rightShift )
{
    int w = static_cast< int >( rawDepth.width() );
    int h = static_cast< int >( rawDepth.height() );

    Array2DReadView< uint16_t > src = rawDepth;
    if( flipX )
    {
        src = libcgt::core::arrayutils::flipX( src );
    }
    if( flipY )
    {
        src = libcgt::core::arrayutils::flipY( src );
    }

    for( int y = 0; y < h; ++y )
    {
        for( int x = 0; x < w; ++x )
        {
            uint16_t d = src[ { x, y } ];
            d >>= rightShift;

            float z = 0.001f * d;
            outputMeters[ { x, y } ] = z;
        }
    }
}

} } } // kinect1x, camera_wrappers, libcgt
