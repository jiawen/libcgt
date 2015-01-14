#pragma once

#include "Vector2f.h"

class QString;
class Rect2i;

class Rect2f
{
public:
	
	Rect2f(); // (0,0,0,0), a null rectangle
	Rect2f( float left, float bottom, float width, float height );
	Rect2f( float width, float height ); // (0, 0, width, height)
	Rect2f( const Vector2f& origin, const Vector2f& size );
	explicit Rect2f( const Vector2f& size ); // (0, 0, width, height)
    explicit Rect2f( const Rect2i& r ); // static_cast int -> float on everything.

	Rect2f( const Rect2f& copy ); // copy constructor
	Rect2f& operator = ( const Rect2f& copy ); // assignment operator

	Vector2f origin() const;
	Vector2f& origin();

	Vector2f size() const;
	Vector2f& size();

    Vector2f limit() const; // origin() + size(); // TODO: better name?

    Vector2f minimum() const; // min( origin, origin + size )
    Vector2f maximum() const; // max( origin, origin + size )

	float width() const; // abs( size.x )
	float height() const; // abs( size.y )
	float area() const; // abs( size.x * size.y )

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
	Rect2f normalized() const;

	QString toString() const;

	// flips this rectangle up/down
	// (usually used to handle rectangles on 2D images where y points down)
    // This rectangle must be standardized.
	Rect2f flippedUD( float height ) const;

    // This rectangle must be standardized.
	// returns the smallest integer-aligned rectangle that contains this
	Rect2i enlargedToInt() const;

	// half-open intervals in x and y
	bool contains( float x, float y );
	bool contains( const Vector2f& point );

    // This rectangle must be standardized.
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
