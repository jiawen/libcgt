#include "KinectUtils.h"

#include <imageproc/Image1f.h>
#include <geometry/Plane3f.h>

// static
Matrix4f KinectUtils::kinectToWorld( const NUI_SKELETON_FRAME& frame )
{
	return worldToKinect( frame ).inverse();
}

// static
Matrix4f KinectUtils::worldToKinect( const NUI_SKELETON_FRAME& frame )
{
	Plane3f plane( frame.vFloorClipPlane.x, frame.vFloorClipPlane.y, frame.vFloorClipPlane.z, frame.vFloorClipPlane.w );

	// x axis is the same
	Vector3f xKinect( 1, 0, 0 );
	// in the Kinect's frame, y is not (0,0,1)
	Vector3f yKinect( frame.vFloorClipPlane.x, frame.vFloorClipPlane.y, frame.vFloorClipPlane.z );
	Vector3f zKinect = Vector3f::cross( xKinect, yKinect );

	// build a world to Kinect matrix
	Matrix4f worldToKinect;	
	worldToKinect.setCol( 0, Vector4f( xKinect, 0 ) );
	// y axis is yKinect
	worldToKinect.setCol( 1, Vector4f( yKinect, 0 ) );
	// z axis is cross product
	worldToKinect.setCol( 2, Vector4f( zKinect, 0 ) );

	// in the world, the Kinect is at the point on the plane closest to the origin
	worldToKinect.setCol( 3, Vector4f( ( plane.pointOnPlane() ), 1 ) );

	return worldToKinect;
}

// static
void KinectUtils::rawDepthMapToMeters( const Array2D< ushort >& rawDepth,
	Image1f& outputMeters, bool flipLeftRight, int rightShft )
{
	int w = rawDepth.width();
	int h = rawDepth.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int xx = x;
			if( flipLeftRight )
			{
				xx = w - x - 1;
			}

			ushort d = rawDepth( x, y );
			d >>= rightShft;

			float z = 0.001f * d;
			outputMeters.setPixel( xx, y, z );
		}
	}
}
