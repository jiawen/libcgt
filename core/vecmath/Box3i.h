#pragma once

#include <string>

#include "libcgt/core/vecmath/Vector3i.h"

class Vector3f;

// A 3D box at integer coordinates, represented as an origin and a size.
//
// All functions treat a box as a Cartesian product of half-open intervals
// [x, x + width) x [y, y + height) x [z, z + depth) and unless otherwise
// marked, functions assume that the box is in "standard form", where all
// components of size are non-negative.
//
// Semantically, the box is defined as right handed: x points right, y points
// up, and z points out of the screen. Therefore, the origin is at coordinates
// "left, bottom, back".
class Box3i
{
public:

    Vector3i origin;
    Vector3i size;

    Box3i() = default;
    explicit Box3i( const Vector3i& size ); // (0, size).
    Box3i( const Vector3i& origin, const Vector3i& size );

    // ------------------------------------------------------------------------
    // These functions only make sense if the box is in standard form and
    // thought of as right handed.
    // ------------------------------------------------------------------------

    int width() const;
    int height() const;
    int depth() const;
    int volume() const;

    int left() const; // origin.x
    int right() const; // origin.x + width
    int bottom() const; // origin.y
    int top() const; // origin.y + height
    int back() const; // origin.z
    int front() const; // origin.z + depth

    Vector3i leftBottomBack() const; // For consistency, same as origin.
    Vector3i rightBottomBack() const;
    Vector3i leftTopBack() const;
    Vector3i rightTopBack() const;
    Vector3i leftBottomFront() const;
    Vector3i rightBottomFront() const;
    Vector3i leftTopFront() const;
    Vector3i rightTopFront() const;

    // ------------------------------------------------------------------------
    // minimum, maximum, and center are well defined even if the box is not in
    // standard form.
    // ------------------------------------------------------------------------

    Vector3i minimum() const; // min( origin, origin + size )
    Vector3i maximum() const; // max( origin, origin + size )

    // TODO: exactCenter()?
    Vector3f center() const;

    // A box is empty if any component of size is zero.
    bool isEmpty() const;

    // Returns true if size() >= 0.
    // Call standardized() to return a valid range with the endpoints flipped
    bool isStandard() const;

    // Returns the same rectangle but with size() >= 0.
    Box3i standardized() const;

    std::string toString() const;

    // flips this box up/down
    // (usually used to handle boxes on 3D images where y points down)
    Box3i flippedUD( int height ) const;

    // flips this box back/front
    // (usually used to handle boxes on 3D volumes where z points in)
    Box3i flippedBF( int depth ) const;

    // half-open intervals in x, y, and z
    bool contains( const Vector3i& p );

    // Returns the smallest Box3i that contains both r0 and r1
    // r0 and r1 must both be valid
    static Box3i united( const Box3i& r0, const Box3i& r1 );

};
