#pragma once

#include "Vector2f.h"

class QString;
class Rect2i;

class Rect2f
{
public:
	
	Rect2f(); // (0,0,0,0), a null rectangle
	Rect2f( float originX, float originY, float width, float height );
	explicit Rect2f( float width, float height ); // (0, 0, width, height)
	Rect2f( const Vector2f& origin, const Vector2f& size );	
	explicit Rect2f( const Vector2f& size ); // (0, 0, width, height)

	Rect2f( const Rect2f& copy ); // copy constructor
	Rect2f& operator = ( const Rect2f& copy ); // assignment operator

	Vector2f origin() const;
	Vector2f& origin();

	Vector2f size() const;
	Vector2f& size();

	float left() const; // origin.x
	float right() const; // origin.x + width

	float bottom() const; // origin.y
	float top() const; // origin.y + height

	Vector2f bottomLeft() const; // for convenience, same as origin
	Vector2f bottomRight() const;
	Vector2f topLeft() const;
	Vector2f topRight() const;

	float width() const;
	float height() const;
	float area() const;

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
	Rect2f flippedUD( float height ) const;

	// returns the smallest integer-aligned rectangle that contains this
	Rect2i enlargedToInt() const;

	// half-open intervals in x and y
	bool contains( float x, float y );
	bool contains( const Vector2f& point );

	// WARNING: the ray is considered a line
	// (tNear and tFar can be < 0)
	bool intersectRay( const Vector2f& origin, const Vector2f& direction,
		float& tNear, float& tFar ) const;

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
