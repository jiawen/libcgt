#pragma once

#include <cassert>
#include "geometry/BoundingBox3f.h"
#include "geometry/Plane3f.h"

#include <vector>
#include "vecmath/Vector3f.h"

class FrustumUtils
{
public:

	enum IntersectionResult
	{
		INSIDE,
		OUTSIDE,
		INTERESECTING
	};

	// *conservatively* tests whether a box intersects a *convex* set of planes defined by a camera frustum
	// (by testing the 8 corners of the box against all 6 planes)
	// 
	// the test is conservative in the sense that:
	//  it's guaranteed to return INSIDE if the box is inside
	//  it's guaranteed to return INTERESECTING if the box straddles the frustum
	//  it *usually* returns OUTSIDE if it's outside, but in some cases, may return INTERSECTING
	static IntersectionResult intersectBoundingBox( const BoundingBox3f& box, Plane3f planes[ 6 ] );

};