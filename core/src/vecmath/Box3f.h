#pragma once

#include <limits>
#include <string>
#include <vector>
#include "vecmath/Vector3f.h"

class Box3i;
class Vector3f;

// A 3D box represented as an origin and a size.
//
// Unless otherwise marked, functions assume that boxes are in "standard form",
// where all components of size are non-negative.
//
// Semantically, the box is defined as right handed: x points right, y points
// up, and z points out of the screen. Therefore, the origin is at coordinates
// "left, bottom, back".
class Box3f
{
public:

    Vector3f origin;
    Vector3f size;

    Box3f() = default;
    explicit Box3f( const Vector3f& size ); // origin = (0, 0, 0)
    Box3f( const Vector3f& origin, const Vector3f& size );

    // ------------------------------------------------------------------------
    // These functions only make sense if the box is in standard form and
    // thought of as right handed.
    // ------------------------------------------------------------------------

    float width() const; // size.x
    float height() const; // size.y
    float depth() const; // size.z
    float volume() const; // size.x * size.y * size.z

    float left() const; // origin.x
    float right() const; // origin.x + width
    float bottom() const; // origin.y
    float top() const; // origin.y + height
    float back() const; // origin.z
    float front() const; // origin.z + depth

    Vector3f leftBottomBack() const; // For consistency, same as origin.
    Vector3f rightBottomBack() const;
    Vector3f leftTopBack() const;
    Vector3f rightTopBack() const;
    Vector3f leftBottomFront() const;
    Vector3f rightBottomFront() const;
    Vector3f leftTopFront() const;
    Vector3f rightTopFront() const;

    // ------------------------------------------------------------------------
    // minimum, maximum, and center are well defined even if the box is not in
    // standard form.
    // ------------------------------------------------------------------------

    Vector3f minimum() const; // min( origin, origin + size )
    Vector3f maximum() const; // max( origin, origin + size )

    Vector3f center() const;

    // A box is empty if any component of size is zero.
    bool isEmpty() const;

    // A box is standard if all components of size are non-negative.
    bool isStandard() const;

    // Returns a standardized version of this box by flipping its coordinates
    // such that all sizes are non-negative.
    Box3f standardized() const;

    std::string toString() const;

    // Flips a standard box left/right.
    Box3f flippedLR( float width ) const;

    // TODO(jiawen): make these free functions in BoxUtilities.
    // Flips a standard box up/down such that it is the same box but the
    // coordinate system points down starting from height. This is usually
    // used to handle 3D volumes where y points down.
    Box3f flippedUD( float height ) const;

    // Flips a standard box back/front.
    Box3f flippedBF( float depth ) const;

    // returns the smallest integer-aligned box that contains this
    Box3i enlargedToInt() const;

    // half-open intervals in x, y, and z
    bool contains( const Vector3f& p );

    void enlargeToContain( const Vector3f& p );

    // Scales the box symmetrically about the center by s[i] along axis i.
    static Box3f scale( const Box3f& b, const Vector3f& s );

    // returns the smallest Box3f that contains both b0 and b1
    // r0 and r1 must both be valid
    static Box3f united( const Box3f& b0, const Box3f& b1 );

    // returns whether two boxes intersect
    static bool intersect( const Box3f& b0, const Box3f& b1 );

    // returns whether two boxes intersect
    // and computes the bounding box that is their intersection
    // (intersection is unmodified if the intersection is empty)
    static bool intersect( const Box3f& b0, const Box3f& b1, Box3f& intersection );

};

// Compute the intersection between a box and a ray, with the ray
// ray parameterized as an origin, direction, and a minimum intersection
// distance (defaulting to 0).
// Calls intersectLine:
//   If both intersections are behind the origin, returns false.
//   Otherwise, sets tIntersect to the closer positive distance.
bool intersectRay( const Box3f& box,
    const Vector3f& rayOrigin, const Vector3f& rayDirection,
    float& tIntersect, float tMin = 0 );

// This function is useful for conservative rasterization and acceleration
// structures.
//
// Carefully compute the intersection between a box and a ray, with the ray
// ray parameterized as an origin, direction, and a minimum intersection
// distance (defaulting to 0).
//
// It also returns which face of the box was hit. 0/1 for -/+ x, 2/3 for -/+ y,
// and 4/5 for -/+ z.
//
// TODO(jiawen): correctly propagate NaN to handle edge on cases.
// TODO(jiawen): maybe a Ray class isn't such a bad idea, with its limits.
bool carefulIntersectBoxRay( const Box3f& box,
    const Vector3f& rayOrigin, const Vector3f& rayDirection,
    float& t0, float& t1, int& t0Face, int& t1Face,
    float rayTMin = 0,
    float rayTMax = std::numeric_limits< float >::infinity() );

// Compute the intersection between a box and a line with the line
// parameterized as an origin and a direction.
// If an intersection is found:
//   Returns true, with the parametric distances tNear and tFar.
//   tNear and tFar can both be < 0. tNear <= tFar.
bool intersectLine( const Box3f& box,
    const Vector3f& rayOrigin, const Vector3f& rayDirection,
    float& tNear, float& tFar );
