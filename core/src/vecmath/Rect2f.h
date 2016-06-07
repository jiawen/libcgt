#pragma once

#include <string>

#include "Vector2f.h"

class Rect2i;

// A 2D rectangle represented as an origin and a size.
//
// Unless otherwise marked, functions assume that boxes are in "standard form",
// where all components of size are non-negative.
//
// Semantically, the rectangle is defined as right handed: x points right
// and y points right. Therefore, the origin is at coordinates "left, bottom".
class Rect2f
{
public:

    Vector2f origin;
    Vector2f size;

    Rect2f() = default;
    explicit Rect2f( const Vector2f& size ); // origin = (0, 0)
    Rect2f( const Vector2f& origin, const Vector2f& size );

    // ------------------------------------------------------------------------
    // These functions only make sense if the rectangle is in standard form and
    // thought of as right handed.
    // ------------------------------------------------------------------------

    float width() const; // size.x
    float height() const; // size.y
    float area() const; // size.x * size.y

    float left() const; // origin.x
    float right() const; // origin.x + width
    float bottom() const; // origin.y
    float top() const; // origin.y + height

    Vector2f leftBottom() const; // For consistency, same as origin.
    Vector2f rightBottom() const;
    Vector2f leftTop() const;
    Vector2f rightTop() const;

    // (size().x, 0)
    Vector2f dx() const;

    // (0, size().y)
    Vector2f dy() const;

    // ------------------------------------------------------------------------
    // minimum, maximum, and center are well defined even if the rectangle is
    // not in standard form.
    // ------------------------------------------------------------------------

    Vector2f minimum() const; // min( origin, origin + size )
    Vector2f maximum() const; // max( origin, origin + size )

    Vector2f center() const;

    // A rectangle is empty if any component of size is zero.
    bool isEmpty() const;

    // Returns true if size.x >= 0 and size.y >= 0.
    // Call standardized() to return a valid range with the endpoints flipped
    bool isStandard() const;

    // Returns the same rectangle but with size() >= 0.
    Rect2f standardized() const;

    std::string toString() const;

    // [Requires a standard rectangle].
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

};
