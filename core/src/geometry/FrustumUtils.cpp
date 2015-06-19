#include "geometry/FrustumUtils.h"

#include "geometry/Plane3f.h"
#include "geometry/BoxUtils.h"
#include "vecmath/Box3f.h"

// static
FrustumUtils::IntersectionResult FrustumUtils::intersectBoundingBox( const Box3f& box, const Plane3f planes[ 6 ] )
{
	auto boxCorners = libcgt::core::geometry::boxutils::corners( box );

	// by default, we assume the box is completely inside
    IntersectionResult result = IntersectionResult::INSIDE;

	// and keep track, for each corner of the box
	// how many vertices are inside vs outside
	int nVerticesInside;
	int nVerticesOutside;

	// for each plane do ...
	for( int i = 0; i < 6; ++i )
	{
		// reset counters for corners nVerticesInside and nVerticesOutside
		nVerticesInside = 0;
		nVerticesOutside = 0;

		// for each corner of the box do ...
		// get nVerticesOutside of the cycle as soon as a box as corners
		// both inside and nVerticesOutside of the frustum
		for( int k = 0; k < 8 && ( nVerticesInside == 0 || nVerticesOutside == 0 ); k++ )
		{
			// is the corner inside or outside?
			float d = planes[ i ].distance( boxCorners[ k ] );
			if( d < 0 )
			{
				++nVerticesInside;
			}
			else
			{
				++nVerticesOutside;
			}
		}

		// if none of the box corners are on the inside halfspace
		// then it's guaranteed to be outside, done
		if( nVerticesInside == 0 )
		{
            return IntersectionResult::OUTSIDE;
		}
		// otherwise, at least some of them are inside
		// but if some of them are *also* outside
		// then we know for now that it intersects this plane
		// (but it's not guaranteed to actually intersect the entire frustum)
		else if( nVerticesOutside != 0 )
		{
            result = IntersectionResult::INTERESECTING;
		}
		// otherwise, we know that some vertices are inside
		// and none are outside
		// --> this box is completely inside (for this plane anyway)
		else
		{
			assert( nVerticesInside == 8 );
			assert( nVerticesOutside == 0 );
		}
	}

	return result;
}
