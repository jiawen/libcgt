#pragma once

#include <initializer_list>

#include "vecmath/Vector3i.h"

class QString;

class Vector3f;

// A 3D box at integer coordinates.
//
// A box is "right handed":
// the z-axis points out of the screen
// so the back has the smallest z coordinate and the front has the largest.
//
// Considered the cartesian product of *half-open* intervals:
// [x, x + width) x [y, y + height) x [z, z + depth)
class Box3i
{
public:

	Box3i(); // (0,0,0,0,0,0), a null box
	explicit Box3i( const Vector3i& size );
	Box3i( const Vector3i& origin, const Vector3i& size );
    // { origin.x, origin.y, origin.z, size.x, size.y, size.z }
    Box3i( std::initializer_list< int > os );

	Box3i( const Box3i& copy ); // copy constructor
	Box3i& operator = ( const Box3i& copy ); // assignment operator

	Vector3i origin() const;
	Vector3i& origin();

	Vector3i size() const;
	Vector3i& size();

    // TODO: make these clear
	int left() const; // origin.x
	int right() const; // origin.x + width

	int bottom() const; // origin.y
	int top() const; // origin.y + height

	int back() const; // origin.z
	int front() const; // origin.z + depth

	Vector3i leftBottomBack() const; // for convenience, same as origin and is considered inside
	Vector3i rightBottomBack() const; // x is one past the end
	Vector3i leftTopBack() const; // y is one past the end
	Vector3i rightTopBack() const; // x and y are one past the end

	Vector3i leftBottomFront() const; // z is one past the end
	Vector3i rightBottomFront() const; // x and z are one past the end
	Vector3i leftTopFront() const; // y and z is one past the end
	Vector3i rightTopFront() const; // x, y, and z are one past the end

	int width() const;
	int height() const;
	int depth() const;
	int volume() const;

    // TODO: exactCenter()?
	Vector3f center() const;

	// Returns true if size() >= 0.
	// Call standardized() to return a valid range with the endpoints flipped
	bool isStandardized() const;

	// Returns the same rectangle but with size() >= 0.
    Box3i standardized() const;

	QString toString() const;

	// flips this box up/down
	// (usually used to handle boxes on 3D images where y points down)
	Box3i flippedUD( int height ) const;

	// flips this box back/front
	// (usually used to handle boxes on 3D volumes where z points in)
	Box3i flippedBF( int depth ) const;

	// flips this box up/down and back/front
	Box3i flippedUDBF( int height, int depth ) const;

	// half-open intervals in x, y, and z
	bool contains( const Vector3i& p );

	// Returns the smallest Box3i that contains both r0 and r1
	// r0 and r1 must both be valid
	static Box3i united( const Box3i& r0, const Box3i& r1 );

private:

	Vector3i m_origin;
	Vector3i m_size;

};
