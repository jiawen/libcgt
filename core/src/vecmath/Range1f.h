#pragma once

#include <string>

#include "Vector2f.h"

class Range1i;

// A 1D range represented as an origin and a size.
//
// Unless otherwise marked, functions assume that the range is in
// "standard form", where size is non-negative.
//
// Semantically, the range is defined with x pofloating right.
class Range1f
{
public:

    float origin = 0;
    float size = 0;

    Range1f() = default;
    explicit Range1f( float size ); // (0, size)
    Range1f( float origin, float size );

    static Range1f fromMinMax( float min, float max );
    static Range1f fromMinMax( const Vector2f& minMax );

    // ------------------------------------------------------------------------
    // These functions only make sense if the range is in standard form.
    // ------------------------------------------------------------------------

    float width() const; // == size, for consistency with Rect and Box.

    float left() const; // origin.x
    float right() const; // origin.x + size
    Vector2f leftRight() const; // ( left(), right() )

    // ------------------------------------------------------------------------
    // minimum, maximum, and center are well defined even if the box is not in
    // standard form.
    // ------------------------------------------------------------------------

    float minimum() const; // min( origin, origin + size )
    float maximum() const; // max( origin, origin + size )

    float center() const;

    // A range is empty if size is zero.
    bool isEmpty() const;

    // A range is standard if size is non-negative.
    bool isStandard() const;

    // Returns a standardized version of this range by flipping its coordinates
    // such that all sizes are non-negative.
    Range1f standardized() const;

    std::string toString() const;

    // [Requires a standard range].
    // Returns the smallest integer-aligned range that contains this.
    Range1i enlargedToInt() const;

    // [Requires a standard range].
    // Returns true if this half-interval contains x.
    bool contains( float x ) const;

    // Returns the smallest Range1f that contains both r0 and r1.
    // r0 and r1 do not have to be standard.
    static Range1f united( const Range1f& r0, const Range1f& r1 );

};
