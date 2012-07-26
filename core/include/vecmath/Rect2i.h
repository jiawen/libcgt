#pragma once

#include "Vector2i.h"

// A 2D rectangle at integer coordinates
class Rect2i
{
public:

	Rect2i( int originX, int originY, int width, int height );
	Rect2i( const Vector2i& origin, const Vector2i& size );
	Rect2i( const Vector2i& size );
	Rect2i( const Rect2i& copy ); // copy constructor
	Rect2i& operator = ( const Rect2i& copy ); // assignment operator

	// TODO: Vector2i& origin() non-const
	// do the same for x() and y() in Vector2f

	Vector2i origin() const;
	Vector2i size() const;
	
	int width() const;
	int height() const;
	int area() const;

	// half-open intervals in x and y
	bool contains( int x, int y );
	bool contains( const Vector2i& p );

private:

	Vector2i m_origin;
	Vector2i m_size;

};
