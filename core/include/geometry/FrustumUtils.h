#pragma once

#include <cassert>

class Box3f;
class Plane3f;

class FrustumUtils
{
public:

	enum class IntersectionResult
	{
		INSIDE,
		OUTSIDE,
		INTERESECTING
	};

	// Conservatively tests whether a box intersects a *convex* set of planes defined by a camera frustum
	// (by testing the 8 corners of the box against all 6 planes).
	// 
	// The test is *conservative* in the sense that:
	//  It's guaranteed to return INSIDE if the box is inside.
	//  It's guaranteed to return INTERESECTING if the box straddles the frustum.
	//  It *usually* returns OUTSIDE if it's outside, but in some cases, may return INTERSECTING.
    static IntersectionResult intersectBoundingBox( const Box3f& box, const Plane3f planes[ 6 ] );

};