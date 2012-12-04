#pragma once

#include "vecmath/Vector3f.h"

class QString;
class Box3i;
class Vector3f;

class Box3f
{
public:

	Box3f(); // (0,0,0,0,0,0), a null box
	Box3f( float left, float bottom, float back, float width, float height, float depth );
	Box3f( float width, float height, float depth );
	Box3f( const Vector3f& origin, const Vector3f& size );
	explicit Box3f( const Vector3f& size );

	Box3f( const Box3f& copy ); // copy constructor
	Box3f& operator = ( const Box3f& copy ); // assignment operator

	Vector3f origin() const;
	Vector3f& origin();

	Vector3f size() const;
	Vector3f& size();

	float left() const; // origin.x
	float right() const; // origin.x + width

	float bottom() const; // origin.y
	float top() const; // origin.y + height

	float back() const; // origin.z
	float front() const; // origin.z + depth

	Vector3f leftBottomBack() const; // for convenience, same as origin and is considered inside
	Vector3f rightBottomBack() const; // x is one past the end
	Vector3f leftTopBack() const; // y is one past the end
	Vector3f rightTopBack() const; // x and y are one past the end

	Vector3f leftBottomFront() const; // z is one past the end
	Vector3f rightBottomFront() const; // x and z are one past the end
	Vector3f leftTopFront() const; // y and z is one past the end
	Vector3f rightTopFront() const; // x, y, and z are one past the end

	float width() const;
	float height() const;
	float depth() const;
	float volume() const;

	Vector3f center() const;

	// returns if this box is null:
	// width == 0 and height == 0
	// (a null box is empty and not valid)
	bool isNull() const;

	// returns true if size().x < 0 or size().y < 0
	// a box is empty iff it's not valid
	// (a null box is empty and not valid)
	// call normalized() to return a valid box with the corners flipped
	bool isEmpty() const;

	// returns true if size().x > 0 and size().y > 0
	// a box is valid iff it's not empty
	// (a null box is empty and not valid)
	// call normalized() to return a valid box with the corners flipped
	bool isValid() const;

	// if this box is invalid,
	// returns a valid box with positive size
	// otherwise, returns this
	// normalizing a null box is still a null box
	Box3f normalized() const;

	QString toString() const;

	// flips this box up/down
	// (usually used to handle boxes on 3D images where y pofloats down)
	Box3f flippedUD( float height ) const;

	// flips this box back/front
	// (usually used to handle boxes on 3D volumes where z pofloats in)
	Box3f flippedBF( float depth ) const;

	// flips this box up/down and back/front
	Box3f flippedUDBF( float height, float depth ) const;
	
	// returns the smallest integer-aligned box that contains this
	Box3i enlargedToInt() const;

	// half-open intervals in x, y, and z
	bool contains( float x, float y, float z );
	bool contains( const Vector3f& p );

	void enlargeToContain( const Vector3f& p );

	// returns the smallest Box3f that contains both b0 and b1
	// r0 and r1 must both be valid
	static Box3f united( const Box3f& b0, const Box3f& b1 );

	// returns whether two boxes intersect
	static bool intersect( const Box3f& b0, const Box3f& b1 );

	// returns whether two boxes intersect
	// and computes the bounding box that is their intersection
	// (intersection is unmodified if the intersection is empty)
	static bool intersect( const Box3f& b0, const Box3f& b1, Box3f& intersection );

private:

	Vector3f m_origin;
	Vector3f m_size;

};
