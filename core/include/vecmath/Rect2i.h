#pragma once

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
	Rect2i( int originX, int originY, int width, int height );
	explicit Rect2i( int width, int height );
	Rect2i( const Vector2i& origin, const Vector2i& size );
	explicit Rect2i( const Vector2i& size );

	Rect2i( const Rect2i& copy ); // copy constructor
	Rect2i& operator = ( const Rect2i& copy ); // assignment operator

	Vector2i origin() const;
	Vector2i& origin();

	Vector2i size() const;
	Vector2i& size();

	int left() const; // origin.x
	int right() const; // origin.x + width

	int bottom() const; // origin.y
	int top() const; // origin.y + height

	Vector2i bottomLeft() const; // for convenience, same as origin and is considered inside
	Vector2i bottomRight() const; // x coordinate is one past the end
	Vector2i topLeft() const; // y coordinate is one past the end
	Vector2i topRight() const; // x and y are both one past the end
	
	int width() const;
	int height() const;
	int area() const;

	Vector2f center() const;

	// returns if this rectangle is null:
	// width == 0 and height == 0
	// (a null rectangle is empty and not valid)
	bool isNull() const;

	// returns true if size().x < 0 or size().y < 0
	// a rectangle is empty iff it's not valid
	// (a null rectangle is empty and not valid)
	// call normalized() to return a valid rectangle with the corners flipped
	bool isEmpty() const;

	// returns true if size().x > 0 and size().y > 0
	// a rectangle is valid iff it's not empty
	// (a null rectangle is empty and not valid)
	// call normalized() to return a valid rectangle with the corners flipped
	bool isValid() const;

	// if this rectangle is invalid,
	// returns a valid rectangle with positive size
	// otherwise, returns this
	// normalizing a null rectangle is still a null rectangle
	Rect2i normalized() const;

	QString toString() const;

	// flips this rectangle up/down
	// (usually used to handle rectangles on 2D images where y points down)
	Rect2i flippedUD( int height ) const;

	// half-open intervals in x and y
	bool contains( int x, int y );
	bool contains( const Vector2i& p );

	// returns the smallest Rect2f that contains both r0 and r1
	// r0 and r1 must both be valid
	static Rect2i united( const Rect2i& r0, const Rect2i& r1 );

private:

	Vector2i m_origin;
	Vector2i m_size;

};
