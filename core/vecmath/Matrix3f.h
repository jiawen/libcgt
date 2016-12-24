#pragma once

class Matrix2f;
class Quat4f;
class Vector2f;

#include <string>
#include "Vector3f.h"

// 3x3 Matrix, stored in column major order (FORTRAN / OpenGL style)
class Matrix3f
{
public:

    // Default is the zero matrix.
    Matrix3f();

    explicit Matrix3f( float fill );

    // Construct 3x3 matrix directly from elements
    // elements are written in row-major order for human-readability
    // (and stored column major, as usual)
    Matrix3f( float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22 );

    // setColumns = true ==> sets the columns of the matrix to be [v0 v1 v2]
    // otherwise, sets the rows
    Matrix3f( const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, bool setColumns = true );

    Matrix3f( const Matrix3f& copy ) = default;
    Matrix3f& operator = ( const Matrix3f& copy ) = default;

    // read / write element (i,j)
    const float& operator () ( int i, int j ) const;
    float& operator () ( int i, int j );

    // get/set row i
    Vector3f getRow( int i ) const;
    void setRow( int i, const Vector3f& v );

    // get/set column j
    Vector3f getCol( int j ) const;
    void setCol( int j, const Vector3f& v );

    // gets the 2x2 submatrix of this matrix to m
    // starting with upper left corner at (i0, j0)
    Matrix2f getSubmatrix2x2( int i0 = 0, int j0 = 0 ) const;

    // sets a 2x2 submatrix of this matrix to m
    // starting with upper left corner at (i0, j0)
    void setSubmatrix2x2( int i0, int j0, const Matrix2f& m );

    float determinant() const;
    Matrix3f inverse() const;
    Matrix3f inverse( bool& isSingular, float epsilon = 0.f ) const; // TODO: invert in place as well

    void transpose();
    Matrix3f transposed() const;

    // ---- Utility ----
    // implicit cast to pointer
    operator const float* () const;
    operator float* ();

    std::string toString() const;

    // uses this to transform a point p (appends a homogeneous coordinate 1, transforms, then extracts xy)
    Vector2f transformPoint( const Vector2f& p ) const;

    // uses this to transform a vector v (appends a homogeneous coordinate 0, transforms, then extracts xy)
    Vector2f transformVector( const Vector2f& v ) const;

    static float determinant3x3( float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22 );

    // ---- Common graphics matrices ----
    static Matrix3f ones();
    static Matrix3f identity();
    static Matrix3f rotateX( float radians );
    static Matrix3f rotateY( float radians );
    static Matrix3f rotateZ( float radians );

    // Assumes axis is normalized.
    static Matrix3f rotation( const Vector3f& axis, float radians );

    // Construct a rotation matrix from an axis-angle representation, where
    // axisAngle.norm() is the rotation angle in radians and
    // axisAngle.normalized() is the direction.
    static Matrix3f rotation( const Vector3f& axisAngle );

    static Matrix3f scaling( const Vector3f& xyz );
    static Matrix3f uniformScaling( float s );
    static Matrix3f translation( const Vector2f& xy );
    // Returns an 2D affine scale-and-translation matrix mapping the rectangle
    // [srcOrigin, srcOrigin + srcSize] --> [dstOrigin, dstOrigin + dstSize]
    static Matrix3f scaleTranslate( const Vector2f& srcOrigin, const Vector2f& srcSize,
        const Vector2f& dstOrigin, const Vector2f& dstSize );

    // Construct a rotation matrix represented by a unit quaternion (does not
    // normalize q).
    static Matrix3f fromQuat( const Quat4f& q );

    // static
    Matrix3f randomRotation( float u0, float u1, float u2 );

    union
    {
        struct
        {
            float m00;
            float m10;
            float m20;

            float m01;
            float m11;
            float m21;

            float m02;
            float m12;
            float m22;
        };
        struct
        {
            Vector3f column0;
            Vector3f column1;
            Vector3f column2;
        };
        float m_elements[ 9 ];
    };

};

Matrix3f operator + ( const Matrix3f& x, const Matrix3f& y );
Matrix3f operator - ( const Matrix3f& x, const Matrix3f& y );
// Negate.
Matrix3f operator - ( const Matrix3f& x );

// Multiply matrix by scalar.
Matrix3f operator * ( float f, const Matrix3f& m );
Matrix3f operator * ( const Matrix3f& m, float f );

// Divide matrix by scalar.
Matrix3f operator / ( const Matrix3f& m, float f );

// Matrix-Vector multiplication
// 3x3 * 3x1 ==> 3x1
Vector3f operator * ( const Matrix3f& m, const Vector3f& v );

// Matrix-Matrix multiplication
Matrix3f operator * ( const Matrix3f& x, const Matrix3f& y );
