#pragma once

#include <vecmath/Matrix4f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>
#include <vector>

class QString;

class BoundingBox3f
{
public:

	// TODO: make a const INFINITY?
	// TODO: make a const ZERO

	// constructs an invalid bounding box with
	// min = numeric_limist< float >.max(),
	// max = numeric_limist< float >.lowest(),
	// so that merge( this, a ) = a
	BoundingBox3f();
	BoundingBox3f( float minX, float minY, float minZ,
		float maxX, float maxY, float maxZ );
	BoundingBox3f( const Vector3f& min, const Vector3f& max );
	BoundingBox3f( const BoundingBox3f& rb );
	BoundingBox3f& operator = ( const BoundingBox3f& rb ); // assignment operator
	// no destructor necessary	

	BoundingBox3f( const std::vector< Vector3f >& points, const Matrix4f& worldMatrix = Matrix4f::identity() );
	BoundingBox3f( const std::vector< Vector4f >& points, const Matrix4f& worldMatrix = Matrix4f::identity() );

	QString toString() const;

	Vector3f& minimum();
	Vector3f& maximum();

	Vector3f minimum() const;
	Vector3f maximum() const;

	// range = maximum - minimum
	Vector3f range() const;

	// center = 0.5f * ( minimum + maximum )
	Vector3f center() const;
	
	float volume() const;

	// returns the minimum of the lengths of the 3 sides of this box
	float shortestSideLength() const;
	
	// returns the maximum of the lengths of the 3 sides of this box
	float longestSideLength() const;

	// returns the 8 corners of the bounding box
	// in the hypercube ordering (x changes most frequently, y next, then z)
	std::vector< Vector3f > corners() const;

	// enlarges the box if p is outside it
	void enlarge( const Vector3f& p );

	// scales the box symmetrically about the center
	void scale( const Vector3f& s );

	// returns true if p is inside this box
	bool containsPoint( const Vector3f& p ) const;

	// TODO: overlaps is the same as intersects??
	// returns if this boundingbox overlaps the other bounding box
	// note that a overlaps b iff b overlaps a
	bool overlaps( const BoundingBox3f& other );

	// Ray is treated as a ray: negative intersections are ignored
	// (tIntersect > 0)
	bool intersectRay( const Vector3f& origin, const Vector3f& direction );
	bool intersectRay( const Vector3f& origin, const Vector3f& direction, float& tIntersect );
	
	// Intersects a ray with full intersections
	// TODO: refactor the above ones to use this one
	bool intersectRay( const Vector3f& origin, const Vector3f& direction, float& tNear, float& tFar ) const;

	// returns the smallest bounding box that contains both bounding boxes
	static BoundingBox3f unite( const BoundingBox3f& b0, const BoundingBox3f& b1 );

    // returns whether two bounding boxes intersect
    static bool intersect( const BoundingBox3f& b0, const BoundingBox3f& b1 );

	// returns whether two bounding boxes intersect
	// and computes the bounding box that is their intersection
	// (intersection is unmodified if the intersection is empty)
	static bool intersect( const BoundingBox3f& b0, const BoundingBox3f& b1, BoundingBox3f& intersection );

private:

	Vector3f m_min;
	Vector3f m_max;

	// TODO: if direction ~ 0, then it's parallel to that slab
	// TODO: early out: http://www.gamedev.net/topic/309689-ray-aabb-intersection/

	// intersects one axis of a ray against a "slab" (interval) defined by [s0,s1]
	// tEnter is updated if the new tEnter is bigger
	// tExit is updated if the new tExit is smaller
	void intersectSlab( float origin, float direction, float s0, float s1,
		float& tEnter, float& tExit );
};
