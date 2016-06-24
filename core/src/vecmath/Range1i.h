#pragma once

#include <string>

#include "Vector2i.h"

// A 1D range at integer coordinates, represented as an origin and a size.
//
// All functions treat a range as a *half-open* interval [x, x + width), and
// unless otherwise marked, functions assume that the range is in
// "standard form", where size is non-negative.
//
// Semantically, the range is defined with x pointing right.
class Range1i
{
public:

    int origin = 0;
    int size = 0;

    Range1i() = default;
    explicit Range1i( int size ); // (0, size)
    Range1i( int origin, int size );

    static Range1i fromMinMax( int min, int max );
    static Range1i fromMinMax( const Vector2i& minMax );

    // ------------------------------------------------------------------------
    // These functions only make sense if the range is in standard form.
    // ------------------------------------------------------------------------

    int width() const; // == size, for consistency with Rect and Box.

    int left() const; // origin.x
    int right() const; // origin.x + size
    Vector2i leftRight() const; // ( left(), right() )

    // ------------------------------------------------------------------------
    // minimum, maximum, and center are well defined even if the box is not in
    // standard form.
    // ------------------------------------------------------------------------

    int minimum() const; // min( origin, origin + size )
    int maximum() const; // max( origin, origin + size )

    // TODO: --> exactCenter()?
    float center() const;

    // A range is empty if size is zero.
    bool isEmpty() const;

    // A range is standard if size is non-negative.
    bool isStandard() const;

    // Returns a standardized version of this range by flipping its coordinates
    // such that all sizes are non-negative.
    Range1i standardized() const;

    std::string toString() const;

    // Whether x is in this half-open interval.
    bool contains( int x );

    // Returns the smallest Range1i that contains both r0 and r1.
    // r0 and r1 do not have to be standard.
    static Range1i united( const Range1i& r0, const Range1i& r1 );

};
