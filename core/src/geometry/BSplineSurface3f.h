#pragma once

#include <vector>

#include "vecmath/Vector2f.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector4f.h"
#include "vecmath/Matrix3f.h"
#include "vecmath/Matrix4f.h"

class BSplineSurface3f
{
public:

    BSplineSurface3f();

    int width() const;
    int height() const;
    Vector2i numControlPoints() const;
    const std::vector< std::vector< Vector3f > >& controlPoints() const;

    // row must be a vector of numControlPoints().x control points
    // increases numControlPoints().y by 1
    void appendControlPointRow( const std::vector< Vector3f >& row );

    // column must be a vector of numControlPoints().y control points
    // increases numControlPoints().x by 1
    void appendControlPointColumn( const std::vector< Vector3f >& column );

    // returns the index of the control point closest to p
    // return -1 if the spline has 0 control points
    // (useful for drag and drop)
    Vector2i controlPointClosestTo( const Vector3f& p, float& distanceSquared ) const;

    // moves control point i to position p
    void moveControlPointTo( const Vector2i& ij, const Vector3f& p );

    Vector3f operator () ( float u, float v ) const;

#if 0

    // getControlPointRow( ... )
    // getControlPointColumn( ... )

    Vector2f evaluateAt( float t ) const;
    Vector2f tangentAt( float t ) const;
    Vector2f normalAt( float t ) const;

    // returns the affine transformation matrix
    // [ nx tx px ]
    // [ ny ty py ]
    // [  0  0  1 ]
    // where (nx,ny) and (tx,ty) are the unit normal and tangent vectors
    Matrix3f frameAt( float t ) const;
#endif

private:

    // outputs a u in [0,1] within each segment
    Vector2i findControlPointStartIndex( float u, float v,
        float& s, float& t ) const;

    Matrix4f m_basis;
    std::vector< std::vector< Vector3f > > m_controlPoints;
};
