#pragma once

#include <initializer_list>
#include <string>

#include "vecmath/Vector2i.h"

class Vector2f;

// A 2D rectangle in integer coordinates.
// Considered the cartesian product of *half-open* intervals:
// [x, x + width) x [y, y + height)
class Rect2i
{
public:

    Rect2i() = default;
    explicit Rect2i( const Vector2i& size ); // origin = (0, 0)
    Rect2i( const Vector2i& origin, const Vector2i& size );
    // { origin.x, origin.y, size.x, size.y }
    Rect2i( std::initializer_list< int > os );

    Rect2i( const Rect2i& copy ) = default;
    Rect2i& operator = ( const Rect2i& copy ) = default;

    // The origin coordinates, as is.
    Vector2i origin() const;
    Vector2i& origin();

    // The size values, as is: they may be negative.
    Vector2i size() const;
    Vector2i& size();

    // origin() + size(),
    // equal to maximum() if this rectangle is standardized,
    // otherwise equal to minimum().
    Vector2i limit() const;

    Vector2i minimum() const; // min( origin, origin + size )
    Vector2i maximum() const; // max( origin, origin + size )

    // (size().x, 0)
    Vector2i dx() const;

    // (0, size().y)
    Vector2i dy() const;

    int width() const; // abs( size.x )
    int height() const; // abs( size.y )
    int area() const; // abs( size.x * size.y )

    // Returns the "center" of this Rect2i in integer coordinates, truncating.
    Vector2i center() const;

    // Returns the "exact center" of this Rect2i in floating point coordinates.
    Vector2f exactCenter() const;

    // Returns true if size() >= 0.
    // Call standardized() to return a valid range with the endpoints flipped
    bool isStandardized() const;

    // Returns the same rectangle but with size() >= 0.
    Rect2i standardized() const;

    std::string toString() const;

    // [Requires a standard rectangle].
    // Returns true if this product of half-open intervals contains p.
    bool contains( const Vector2i& p );

    // Returns the smallest Rect2i that contains both r0 and r1.
    // r0 and r1 must both be valid.
    static Rect2i united( const Rect2i& r0, const Rect2i& r1 );

private:

    Vector2i m_origin;
    Vector2i m_size;

};
