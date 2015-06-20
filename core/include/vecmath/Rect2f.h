#pragma once

#include <initializer_list>
#include <string>

#include "Vector2f.h"

class QString;
class Rect2i;

// A 2D rectangle in floating point coordinates.
// Considered the cartesian product of *half-open* intervals:
// [x, x + width) x [y, y + height)
class Rect2f
{
public:

    // The default constructor makes the null rectangle with origin and size all 0.
    Rect2f() = default;
    explicit Rect2f( const Vector2f& size ); // origin = (0, 0)
    Rect2f( const Vector2f& origin, const Vector2f& size );
    // { origin.x, origin.y, size.x, size.y }
    Rect2f( std::initializer_list< float > os );

    explicit Rect2f( const Rect2i& r ); // static_cast int -> float on everything.

    Rect2f( const Rect2f& copy ) = default;
    Rect2f& operator = ( const Rect2f& copy ) = default;

    // The origin coordinates, as is.
    Vector2f origin() const;
    Vector2f& origin();

    // The size values, as is: they may be negative.
    Vector2f size() const;
    Vector2f& size();

    // origin() + size(),
    // equal to maximum() if this rectangle is standardized,
    // otherwise equal to minimum().
    Vector2f limit() const;

    Vector2f minimum() const; // min( origin, origin + size )
    Vector2f maximum() const; // max( origin, origin + size )

    // (size().x, 0)
    Vector2f dx() const;

    // (0, size().y)
    Vector2f dy() const;

    float width() const; // abs( size.x )
    float height() const; // abs( size.y )
    float area() const; // abs( size.x * size.y )

    Vector2f center() const;

    // Returns true if size() >= 0.
    // Call standardized() to return a valid range with the endpoints flipped
    bool isStandardized() const;

    // Returns the same rectangle but with size() >= 0.
    Rect2f standardized() const;

    std::string toString() const;

    // This rectangle must be standardized.
    // Returns the smallest integer-aligned rectangle that contains this.
    Rect2i enlargedToInt() const;

    // [Requires a standard rectangle].
    // Returns true if this product of half-open intervals contains p.
    bool contains( const Vector2f& p ) const;

    // [Requires a standard rectangle].
    // WARNING: the ray is considered a line
    // (tNear and tFar can be < 0)
    // returns true if the ray intersects this rectangle
    //   tNear (< tFar) is the first intersection
    //   tFar is the second intersection
    //   axis is 0 if tNear hit the left or right edge (x-axis)
    //   axis is 1 if tNear hit the bottom or top edge (y-axis)
    bool intersectRay( const Vector2f& rayOrigin, const Vector2f& rayDirection,
        float& tNear, float& tFar, int& axis ) const;

    // returns the smallest Rect2f that contains both r0 and r1
    // r0 and r1 must both be valid
    static Rect2f united( const Rect2f& r0, const Rect2f& r1 );

    // returns whether two rectangles intersect
    static bool intersect( const Rect2f& r0, const Rect2f& r1 );

    // returns whether two rectangles intersect
    // and computes the bounding box that is their intersection
    // (intersection is unmodified if the intersection is empty)
    static bool intersect( const Rect2f& r0, const Rect2f& r1, Rect2f& intersection );

private:

    Vector2f m_origin;
    Vector2f m_size;

};
