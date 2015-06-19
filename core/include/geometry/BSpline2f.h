#pragma once

#include <vector>

#include "vecmath/Vector2f.h"
#include "vecmath/Vector4f.h"
#include "vecmath/Matrix3f.h"
#include "vecmath/Matrix4f.h"

class BSpline2f
{
public:

	BSpline2f();

	int numControlPoints() const;
	const std::vector< Vector2f >& controlPoints() const;

	void appendControlPoint( const Vector2f& p );

	// returns the index of the control point closest to p
	// return -1 if the spline has 0 control points
	// (useful for drag and drop)
	int controlPointClosestTo( const Vector2f& p, float& distanceSquared ) const;

	// moves control point i to position p
	void moveControlPointTo( int i, const Vector2f& p );
	

	Vector2f operator [] ( float t ) const;

	Vector2f evaluateAt( float t ) const;
	Vector2f tangentAt( float t ) const;
	Vector2f normalAt( float t ) const;

	// returns the affine transformation matrix
	// [ nx tx px ]
	// [ ny ty py ]
	// [  0  0  1 ]
	// where (nx,ny) and (tx,ty) are the unit normal and tangent vectors
	Matrix3f frameAt( float t ) const;

private:

	// outputs a u in [0,1] within each segment
	int findControlPointStartIndex( float t, float& u ) const;

	Matrix4f m_basis;
	std::vector< Vector2f > m_controlPoints;
};
