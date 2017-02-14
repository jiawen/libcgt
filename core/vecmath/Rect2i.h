#pragma once

#include <string>

#include "libcgt/core/vecmath/Vector2i.h"

class Vector2f;
class Rect2f;

// A 2D rectangle at integer coordinates, represented as an origin and a size.
//
// All functions treat a box as a Cartesian product of half-open intervals
// [x, x + width) x [y, y + height) and unless otherwise marked, functions
// assume that the box is in "standard form", where all components of size are
// non-negative.
//
// Semantically, the rectangle is defined as right handed: x points right, and
// y points right. Therefore, the origin is at coordinates "left, bottom".
class Rect2i
{
public:

    Vector2i origin;
    Vector2i size;

    Rect2i() = default;
    explicit Rect2i( const Vector2i& size ); // origin = (0, 0)
    Rect2i( const Vector2i& origin, const Vector2i& size );

    // ------------------------------------------------------------------------
    // These functions only make sense if the rectangle is in standard form and
    // thought of as right handed.
    // ------------------------------------------------------------------------

    int width() const;
    int height() const;
    int area() const;

    int left() const; // origin.x
    int right() const; // origin.x + width
    int bottom() const; // origin.y
    int top() const; // origin.y + height

    Vector2i leftBottom() const; // For consistency, same as origin.
    Vector2i rightBottom() const;
    Vector2i leftTop() const;
    Vector2i rightTop() const;

    // (size().x, 0)
    Vector2i dx() const;

    // (0, size().y)
    Vector2i dy() const;

    // ------------------------------------------------------------------------
    // minimum, maximum, and center are well defined even if the rectangle is
    // not in standard form.
    // ------------------------------------------------------------------------

    Vector2i minimum() const; // min( origin, origin + size )
    Vector2i maximum() const; // max( origin, origin + size )

    // Returns the "center" of this Rect2i in integer coordinates, truncating.
    Vector2i center() const;

    // Returns the "exact center" of this Rect2i in floating point coordinates.
    Vector2f exactCenter() const;

    // A rectangle is empty if any component of size is zero.
    bool isEmpty() const;

    // Returns true if size() >= 0.
    // Call standardized() to return a valid range with the endpoints flipped
    bool isStandard() const;

    // Returns the same rectangle but with size() >= 0.
    Rect2i standardized() const;

    std::string toString() const;

    // [Requires a standard rectangle].
    // Returns true if this product of half-open intervals contains p.
    bool contains( const Vector2i& p );

    Rect2f castToFloat() const;

    // Returns the smallest Rect2i that contains both r0 and r1.
    // r0 and r1 must both be valid.
    static Rect2i united( const Rect2i& r0, const Rect2i& r1 );

};
