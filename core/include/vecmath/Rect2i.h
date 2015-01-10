#pragma once

#include <initializer_list>

#include "vecmath/Vector2i.h"

class QString;

class Vector2f;

// A 2D rectangle at integer coordinates
// Considered the cartesian product of *half-open* intervals:
// [x, x + width) x [y, y + height)
class Rect2i
{
public:

	Rect2i(); // (0,0,0,0), a null rectangle
	explicit Rect2i( const Vector2i& size ); // origin = (0, 0)
	Rect2i( const Vector2i& origin, const Vector2i& size );
    // { origin.x, origin.y, size.x, size.y }
    Rect2i( std::initializer_list< int > os );

	Rect2i( const Rect2i& copy ); // copy constructor
	Rect2i& operator = ( const Rect2i& copy ); // assignment operator

    // The origin coordinates, as is.
	Vector2i origin() const;
	Vector2i& origin();

    // The size values, as is: they may be negative.
	Vector2i size() const;
	Vector2i& size();

    // TODO: make these clear
    // minimum(), maximum() --> don't have coordinate conventions
    // and clearly works on on-standard
    // standardized --> isStandard(), get rid of left, right, ...
	int left() const; // origin.x
	int right() const; // origin.x + size.x

	int bottom() const; // origin.y
	int top() const; // origin.y + size.y

	Vector2i bottomLeft() const; // for convenience, same as origin and is considered inside
	Vector2i bottomRight() const; // x coordinate is one past the end
	Vector2i topLeft() const; // y coordinate is one past the end
	Vector2i topRight() const; // x and y are both one past the end
	
	int width() const;
	int height() const;
	int area() const;

	Vector2f center() const;

	// Returns true if size() >= 0.
	// Call standardized() to return a valid range with the endpoints flipped
	bool isStandardized() const;

	// Returns the same rectangle but with size() >= 0.
	Rect2i standardized() const;

	QString toString() const;

	// (Requires a standard rectangle)
    // Returns the exact same rectangle as this but in a coordinate system
    // that counts from [0, height) but where y points down.
	// (usually used to handle rectangles on 2D images where y points down).
	Rect2i flippedUD( int height ) const;

	// half-open intervals in x and y
	bool contains( const Vector2i& p );

	// Returns the smallest Rect2i that contains both r0 and r1.
	// r0 and r1 must both be valid.
	static Rect2i united( const Rect2i& r0, const Rect2i& r1 );

private:

	Vector2i m_origin;
	Vector2i m_size;

};
