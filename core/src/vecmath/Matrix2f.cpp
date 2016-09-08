#include "vecmath/Matrix2f.h"

#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>

using std::abs;

Matrix2f::Matrix2f() :
    Matrix2f( 0.0f )
{

}

Matrix2f::Matrix2f( float fill )
{
    for( int i = 0; i < 4; ++i )
    {
        m_elements[ i ] = fill;
    }
}

Matrix2f::Matrix2f( float m00, float m01,
                   float m10, float m11 )
{
    m_elements[ 0 ] = m00;
    m_elements[ 1 ] = m10;
    m_elements[ 2 ] = m01;
    m_elements[ 3 ] = m11;
}

Matrix2f::Matrix2f( const Vector2f& v0, const Vector2f& v1, bool setColumns )
{
    if( setColumns )
    {
        setCol( 0, v0 );
        setCol( 1, v1 );
    }
    else
    {
        setRow( 0, v0 );
        setRow( 1, v1 );
    }
}

const float& Matrix2f::operator () ( int i, int j ) const
{
    return m_elements[ j * 2 + i ];
}

float& Matrix2f::operator () ( int i, int j )
{
    return m_elements[ j * 2 + i ];
}

Vector2f Matrix2f::getRow( int i ) const
{
    return
    {
        m_elements[ i ],
        m_elements[ i + 2 ]
    };
}

void Matrix2f::setRow( int i, const Vector2f& v )
{
    m_elements[ i ] = v.x;
    m_elements[ i + 2 ] = v.y;
}

Vector2f Matrix2f::getCol( int j ) const
{
    int colStart = 2 * j;

    return
    {
        m_elements[ colStart ],
        m_elements[ colStart + 1 ]
    };
}

void Matrix2f::setCol( int j, const Vector2f& v )
{
    int colStart = 2 * j;

    m_elements[ colStart ] = v.x;
    m_elements[ colStart + 1 ] = v.y;
}

float Matrix2f::determinant() const
{
    return Matrix2f::determinant2x2
    (
        m_elements[ 0 ], m_elements[ 2 ],
        m_elements[ 1 ], m_elements[ 3 ]
    );
}

Matrix2f Matrix2f::inverse() const
{
    bool isSingular;
    return inverse( isSingular );
}

Matrix2f Matrix2f::inverse( bool& isSingular, float epsilon ) const
{
    float det = determinant();

    isSingular = ( abs( det ) < epsilon );
    if( isSingular )
    {
        return Matrix2f();
    }
    else
    {
        float reciprocalDeterminant = 1.0f / det;

        return Matrix2f
        (
            m_elements[ 3 ] * reciprocalDeterminant, -m_elements[ 2 ] * reciprocalDeterminant,
            -m_elements[ 1 ] * reciprocalDeterminant, m_elements[ 0 ] * reciprocalDeterminant
        );
    }
}

void Matrix2f::transpose()
{
    float _m01 = m01;
    float _m10 = m10;

    m01 = _m10;
    m10 = _m01;
}

Matrix2f Matrix2f::transposed() const
{
    return Matrix2f
    (
        m00, m10,
        m01, m11
    );
}

Matrix2f::operator const float* () const
{
    return m_elements;
}

Matrix2f::operator float* ()
{
    return m_elements;
}

std::string Matrix2f::toString() const
{
    const int FIELD_WIDTH = 8;
    const int PRECISION = 4;

    std::ostringstream sstream;
    sstream << std::fixed << std::setprecision( PRECISION ) <<
        std::setiosflags( std::ios::right );
    sstream << std::endl;

    for( int i = 0; i < 2; ++i )
    {
        sstream << "[ ";
        for( int j = 0; j < 2; ++j )
        {
            sstream << std::setw( FIELD_WIDTH ) << ( *this )( i, j ) << " ";
        }
        sstream << "]" << std::endl;
    }
    return sstream.str();
}

// static
float Matrix2f::determinant2x2( float m00, float m01,
                               float m10, float m11 )
{
    return( m00 * m11 - m01 * m10 );
}

// static
Matrix2f Matrix2f::ones()
{
    Matrix2f m;
    for( int i = 0; i < 4; ++i )
    {
        m.m_elements[ i ] = 1;
    }

    return m;
}

// static
Matrix2f Matrix2f::identity()
{
    Matrix2f m;

    m( 0, 0 ) = 1;
    m( 1, 1 ) = 1;

    return m;
}

// static
Matrix2f Matrix2f::rotation( float degrees )
{
    float c = cos( degrees );
    float s = sin( degrees );

    return Matrix2f
    (
        c, -s,
        s, c
    );
}

//////////////////////////////////////////////////////////////////////////
// Operators
//////////////////////////////////////////////////////////////////////////

Matrix2f operator + ( const Matrix2f& x, const Matrix2f& y )
{
    Matrix2f sum;

    for( int k = 0; k < 4; ++k )
    {
        sum[k] = x[k] + y[k];
    }

    return sum;
}

Matrix2f operator - ( const Matrix2f& x, const Matrix2f& y )
{
    Matrix2f difference;

    for( int k = 0; k < 4; ++k )
    {
        difference[k] = x[k] - y[k];
    }

    return difference;
}

Matrix2f operator - ( const Matrix2f& x )
{
    Matrix2f output;

    for( int k = 0; k < 4; ++k )
    {
        output[k] = -x[k];
    }

    return output;
}

Matrix2f operator * ( float f, const Matrix2f& m )
{
    Matrix2f output;

    for( int k = 0; k < 4; ++k )
    {
        output[k] = f * m[k];
    }

    return output;
}

Matrix2f operator * ( const Matrix2f& m, float f )
{
    return f * m;
}

Matrix2f operator / ( const Matrix2f& m, float f )
{
    return ( 1.0f / f ) * m;
}

Vector2f operator * ( const Matrix2f& m, const Vector2f& v )
{
    Vector2f output( 0 );

    for( int i = 0; i < 2; ++i )
    {
        for( int j = 0; j < 2; ++j )
        {
            output[ i ] += m( i, j ) * v[ j ];
        }
    }

    return output;
}

Matrix2f operator * ( const Matrix2f& x, const Matrix2f& y )
{
    Matrix2f product; // zeroes

    for( int i = 0; i < 2; ++i )
    {
        for( int j = 0; j < 2; ++j )
        {
            for( int k = 0; k < 2; ++k )
            {
                product( i, k ) += x( i, j ) * y( j, k );
            }
        }
    }

    return product;
}
