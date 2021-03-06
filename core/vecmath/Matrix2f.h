#pragma once

#include <string>
#include "libcgt/core/vecmath/Vector2f.h"

// 2x2 Matrix, stored in column major order (OpenGL style)
class Matrix2f
{
public:

    // Default is the zero matrix.
    Matrix2f();

    explicit Matrix2f( float fill );

    Matrix2f( float m00, float m01,
        float m10, float m11 );

    // setColumns = true ==> sets the columns of the matrix to be [v0 v1]
    // otherwise, sets the rows
    Matrix2f( const Vector2f& v0, const Vector2f& v1, bool setColumns = true );

    Matrix2f( const Matrix2f& copy ) = default;
    Matrix2f& operator = ( const Matrix2f& rm ) = default;

    const float& operator () ( int i, int j ) const;
    float& operator () ( int i, int j );

    Vector2f getRow( int i ) const;
    void setRow( int i, const Vector2f& v );

    Vector2f getCol( int j ) const;
    void setCol( int j, const Vector2f& v );

    float determinant() const;
    Matrix2f inverse() const;
    Matrix2f inverse( bool& isSingular, float epsilon = 0.f ) const; // TODO: in place inverse

    void transpose();
    Matrix2f transposed() const;

    // implicit cast
    operator const float* () const;
    operator float* ();

    std::string toString() const;

    static float determinant2x2( float m00, float m01,
        float m10, float m11 );

    static Matrix2f ones();
    static Matrix2f identity();
    static Matrix2f rotation( float degrees );

    union
    {
        struct
        {
            float m00;
            float m10;

            float m01;
            float m11;
        };
        struct
        {
            Vector2f column0;
            Vector2f column1;
        };
        float m_elements[ 4 ];
    };
};

Matrix2f operator + ( const Matrix2f& x, const Matrix2f& y );
Matrix2f operator - ( const Matrix2f& x, const Matrix2f& y );
// Negate.
Matrix2f operator - ( const Matrix2f& x );

// Multiply matrix by scalar.
Matrix2f operator * ( float f, const Matrix2f& m );
Matrix2f operator * ( const Matrix2f& m, float f );

// Divide matrix by scalar.
Matrix2f operator / ( const Matrix2f& m, float f );

// Matrix-Vector multiplication
// 2x2 * 2x1 ==> 2x1
Vector2f operator * ( const Matrix2f& m, const Vector2f& v );

// Matrix-Matrix multiplication
Matrix2f operator * ( const Matrix2f& x, const Matrix2f& y );
