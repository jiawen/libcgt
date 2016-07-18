#include "vecmath/Matrix4f.h"

#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>

#include <math/MathUtils.h>

#include "vecmath/Matrix2f.h"
#include "vecmath/Matrix3f.h"
#include "vecmath/Quat4f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Rect2f.h"

using std::abs;
using libcgt::core::math::cot;

// static
Matrix4f Matrix4f::ROTATE_X_180 = Matrix4f
(
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1
);

// static
Matrix4f Matrix4f::ROTATE_Y_180 = Matrix4f
(
    -1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1
);

// static
Matrix4f Matrix4f::ROTATE_Z_180 = Matrix4f
(
    -1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
);

Matrix4f::Matrix4f() :
    Matrix4f( 0.0f )
{
}

Matrix4f::Matrix4f( float fill )
{
    for( int i = 0; i < 16; ++i )
    {
        m_elements[ i ] = fill;
    }
}

Matrix4f::Matrix4f( float m00, float m01, float m02, float m03,
                   float m10, float m11, float m12, float m13,
                   float m20, float m21, float m22, float m23,
                   float m30, float m31, float m32, float m33 )
{
    m_elements[ 0 ] = m00;
    m_elements[ 1 ] = m10;
    m_elements[ 2 ] = m20;
    m_elements[ 3 ] = m30;

    m_elements[ 4 ] = m01;
    m_elements[ 5 ] = m11;
    m_elements[ 6 ] = m21;
    m_elements[ 7 ] = m31;

    m_elements[ 8 ] = m02;
    m_elements[ 9 ] = m12;
    m_elements[ 10 ] = m22;
    m_elements[ 11 ] = m32;

    m_elements[ 12 ] = m03;
    m_elements[ 13 ] = m13;
    m_elements[ 14 ] = m23;
    m_elements[ 15 ] = m33;
}

Matrix4f::Matrix4f( const Vector4f& v0, const Vector4f& v1, const Vector4f& v2, const Vector4f& v3, bool setColumns )
{
    if( setColumns )
    {
        setCol( 0, v0 );
        setCol( 1, v1 );
        setCol( 2, v2 );
        setCol( 3, v3 );
    }
    else
    {
        setRow( 0, v0 );
        setRow( 1, v1 );
        setRow( 2, v2 );
        setRow( 3, v3 );
    }
}

const float& Matrix4f::operator () ( int i, int j ) const
{
    return m_elements[ j * 4 + i ];
}

float& Matrix4f::operator () ( int i, int j )
{
    return m_elements[ j * 4 + i ];
}

Vector4f Matrix4f::getRow( int i ) const
{
    return Vector4f
    (
        m_elements[ i ],
        m_elements[ i + 4 ],
        m_elements[ i + 8 ],
        m_elements[ i + 12 ]
    );
}

void Matrix4f::setRow( int i, const Vector4f& v )
{
    m_elements[ i ] = v.x;
    m_elements[ i + 4 ] = v.y;
    m_elements[ i + 8 ] = v.z;
    m_elements[ i + 12 ] = v.w;
}

Vector4f Matrix4f::getCol( int j ) const
{
    int colStart = 4 * j;

    return Vector4f
    (
        m_elements[ colStart ],
        m_elements[ colStart + 1 ],
        m_elements[ colStart + 2 ],
        m_elements[ colStart + 3 ]
    );
}

void Matrix4f::setCol( int j, const Vector4f& v )
{
    int colStart = 4 * j;

    m_elements[ colStart ] = v.x;
    m_elements[ colStart + 1 ] = v.y;
    m_elements[ colStart + 2 ] = v.z;
    m_elements[ colStart + 3 ] = v.w;
}

Matrix2f Matrix4f::getSubmatrix2x2( int i0, int j0 ) const
{
    Matrix2f out;

    for( int i = 0; i < 2; ++i )
    {
        for( int j = 0; j < 2; ++j )
        {
            out( i, j ) = ( *this )( i + i0, j + j0 );
        }
    }

    return out;
}

Matrix3f Matrix4f::getSubmatrix3x3( int i0, int j0 ) const
{
    Matrix3f out;

    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            out( i, j ) = ( *this )( i + i0, j + j0 );
        }
    }

    return out;
}

void Matrix4f::setSubmatrix2x2( int i0, int j0, const Matrix2f& m )
{
    for( int i = 0; i < 2; ++i )
    {
        for( int j = 0; j < 2; ++j )
        {
            ( *this )( i + i0, j + j0 ) = m( i, j );
        }
    }
}

void Matrix4f::setSubmatrix3x3( int i0, int j0, const Matrix3f& m )
{
    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            ( *this )( i + i0, j + j0 ) = m( i, j );
        }
    }
}

float Matrix4f::determinant() const
{
    float cofactor00 =  Matrix3f::determinant3x3( m11, m12, m13, m21, m22, m23, m31, m32, m33 );
    float cofactor01 = -Matrix3f::determinant3x3( m12, m13, m10, m22, m23, m20, m32, m33, m30 );
    float cofactor02 =  Matrix3f::determinant3x3( m13, m10, m11, m23, m20, m21, m33, m30, m31 );
    float cofactor03 = -Matrix3f::determinant3x3( m10, m11, m12, m20, m21, m22, m30, m31, m32 );

    return( m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02 + m03 * cofactor03 );
}

Matrix4f Matrix4f::inverse() const
{
    bool singular;
    return inverse( singular );
}

Matrix4f Matrix4f::inverse( bool& isSingular, float epsilon ) const
{
    float cofactor00 =  Matrix3f::determinant3x3( m11, m12, m13, m21, m22, m23, m31, m32, m33 );
    float cofactor01 = -Matrix3f::determinant3x3( m12, m13, m10, m22, m23, m20, m32, m33, m30 );
    float cofactor02 =  Matrix3f::determinant3x3( m13, m10, m11, m23, m20, m21, m33, m30, m31 );
    float cofactor03 = -Matrix3f::determinant3x3( m10, m11, m12, m20, m21, m22, m30, m31, m32 );

    float cofactor10 = -Matrix3f::determinant3x3( m21, m22, m23, m31, m32, m33, m01, m02, m03 );
    float cofactor11 =  Matrix3f::determinant3x3( m22, m23, m20, m32, m33, m30, m02, m03, m00 );
    float cofactor12 = -Matrix3f::determinant3x3( m23, m20, m21, m33, m30, m31, m03, m00, m01 );
    float cofactor13 =  Matrix3f::determinant3x3( m20, m21, m22, m30, m31, m32, m00, m01, m02 );

    float cofactor20 =  Matrix3f::determinant3x3( m31, m32, m33, m01, m02, m03, m11, m12, m13 );
    float cofactor21 = -Matrix3f::determinant3x3( m32, m33, m30, m02, m03, m00, m12, m13, m10 );
    float cofactor22 =  Matrix3f::determinant3x3( m33, m30, m31, m03, m00, m01, m13, m10, m11 );
    float cofactor23 = -Matrix3f::determinant3x3( m30, m31, m32, m00, m01, m02, m10, m11, m12 );

    float cofactor30 = -Matrix3f::determinant3x3( m01, m02, m03, m11, m12, m13, m21, m22, m23 );
    float cofactor31 =  Matrix3f::determinant3x3( m02, m03, m00, m12, m13, m10, m22, m23, m20 );
    float cofactor32 = -Matrix3f::determinant3x3( m03, m00, m01, m13, m10, m11, m23, m20, m21 );
    float cofactor33 =  Matrix3f::determinant3x3( m00, m01, m02, m10, m11, m12, m20, m21, m22 );

    float determinant = m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02 + m03 * cofactor03;

    isSingular = ( abs( determinant ) < epsilon );
    if( isSingular )
    {
        return Matrix4f();
    }
    else
    {
        float reciprocalDeterminant = 1.0f / determinant;

        return Matrix4f
        (
            cofactor00 * reciprocalDeterminant, cofactor10 * reciprocalDeterminant, cofactor20 * reciprocalDeterminant, cofactor30 * reciprocalDeterminant,
            cofactor01 * reciprocalDeterminant, cofactor11 * reciprocalDeterminant, cofactor21 * reciprocalDeterminant, cofactor31 * reciprocalDeterminant,
            cofactor02 * reciprocalDeterminant, cofactor12 * reciprocalDeterminant, cofactor22 * reciprocalDeterminant, cofactor32 * reciprocalDeterminant,
            cofactor03 * reciprocalDeterminant, cofactor13 * reciprocalDeterminant, cofactor23 * reciprocalDeterminant, cofactor33 * reciprocalDeterminant
        );
    }
}

void Matrix4f::transpose()
{
    float temp;

    for( int i = 0; i < 3; ++i )
    {
        for( int j = i + 1; j < 4; ++j )
        {
            temp = ( *this )( i, j );
            ( * this )( i, j ) = ( *this )( j, i );
            ( *this )( j, i ) = temp;
        }
    }
}

Matrix4f Matrix4f::transposed() const
{
    Matrix4f out;
    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            out( j, i ) = ( *this )( i, j );
        }
    }

    return out;
}

Matrix3f Matrix4f::normalMatrix() const
{
    return getSubmatrix3x3().inverse().transposed();
}

Matrix4f Matrix4f::normalMatrix4x4() const
{
    Matrix3f n = getSubmatrix3x3().inverse().transposed();
    Matrix4f n4;
    n4.setSubmatrix3x3( 0, 0, n );
    return n4;
}

Matrix4f::operator const float* () const
{
    return m_elements;
}

Matrix4f::operator float* ()
{
    return m_elements;
}

std::string Matrix4f::toString() const
{
    const int kFieldWidth = 8;
    const int kPrecision = 4;

    std::ostringstream sstream;
    sstream << std::fixed << std::setprecision( kPrecision ) <<
        std::setiosflags( std::ios::right );
    sstream << std::endl;

    for( int i = 0; i < 4; ++i )
    {
        sstream << "[ ";
        for( int j = 0; j < 4; ++j )
        {
            sstream << std::setw( kFieldWidth ) << ( *this )( i, j ) << " ";
        }
        sstream << "]" << std::endl;
    }
    return sstream.str();
}

Vector3f Matrix4f::transformPoint( const Vector3f& p ) const
{
    return ( ( *this ) * Vector4f( p, 1 ) ).xyz;
}

Vector3f Matrix4f::transformVector( const Vector3f& v ) const
{
    return ( ( *this ) * Vector4f( v, 0 ) ).xyz;
}

Vector3f Matrix4f::transformNormal( const Vector3f& n ) const
{
    return normalMatrix() * n;
}

// static
Matrix4f Matrix4f::ones()
{
    Matrix4f m;
    for( int i = 0; i < 16; ++i )
    {
        m.m_elements[ i ] = 1;
    }

    return m;
}

// static
Matrix4f Matrix4f::identity()
{
    Matrix4f m;

    m( 0, 0 ) = 1;
    m( 1, 1 ) = 1;
    m( 2, 2 ) = 1;
    m( 3, 3 ) = 1;

    return m;
}

// static
Matrix4f Matrix4f::translation( const Vector3f& xyz )
{
    return Matrix4f
    (
        1, 0, 0, xyz.x,
        0, 1, 0, xyz.y,
        0, 0, 1, xyz.z,
        0, 0, 0, 1
    );
}

// static
Matrix4f Matrix4f::rotateX( float radians )
{
    float c = cos( radians );
    float s = sin( radians );

    return Matrix4f
    (
        1, 0, 0, 0,
        0, c, -s, 0,
        0, s, c, 0,
        0, 0, 0, 1
    );
}

// static
Matrix4f Matrix4f::rotateY( float radians )
{
    float c = cos( radians );
    float s = sin( radians );

    return Matrix4f
    (
        c, 0, s, 0,
        0, 1, 0, 0,
        -s, 0, c, 0,
        0, 0, 0, 1
    );
}

// static
Matrix4f Matrix4f::rotateZ( float radians )
{
    float c = cos( radians );
    float s = sin( radians );

    return Matrix4f
    (
        c, -s, 0, 0,
        s, c, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );
}

// static
Matrix4f Matrix4f::rotation( const Vector3f& axis, float radians )
{
    if( axis.normSquared() == 0 || radians == 0 )
    {
        return Matrix4f::identity();
    }

    Vector3f normalizedDirection = axis.normalized();

    float cosTheta = cos( radians );
    float sinTheta = sin( radians );

    float x = normalizedDirection.x;
    float y = normalizedDirection.y;
    float z = normalizedDirection.z;

    return Matrix4f
    (
        x * x * ( 1.0f - cosTheta ) + cosTheta,         y * x * ( 1.0f - cosTheta ) - z * sinTheta,     z * x * ( 1.0f - cosTheta ) + y * sinTheta,     0.0f,
        x * y * ( 1.0f - cosTheta ) + z * sinTheta,     y * y * ( 1.0f - cosTheta ) + cosTheta,         z * y * ( 1.0f - cosTheta ) - x * sinTheta,     0.0f,
        x * z * ( 1.0f - cosTheta ) - y * sinTheta,     y * z * ( 1.0f - cosTheta ) + x * sinTheta,     z * z * ( 1.0f - cosTheta ) + cosTheta,         0.0f,
        0.0f,                                           0.0f,                                           0.0f,                                           1.0f
    );
}

// static
Matrix4f Matrix4f::scaling( const Vector3f& xyz )
{
    return Matrix4f
    (
        xyz.x, 0, 0, 0,
        0, xyz.y, 0, 0,
        0, 0, xyz.z, 0,
        0, 0, 0, 1
    );
}

// static
Matrix4f Matrix4f::uniformScaling( float s )
{
    return Matrix4f
    (
        s, 0, 0, 0,
        0, s, 0, 0,
        0, 0, s, 0,
        0, 0, 0, 1
    );
}

// static
Matrix4f Matrix4f::scaleTranslate( const Vector3f& srcOrigin, const Vector3f& srcSize,
    const Vector3f& dstOrigin, const Vector3f& dstSize )
{
    // translate rectangle to have its origin at (0,0)
    Matrix4f t0 = Matrix4f::translation( -srcOrigin );

    // scale it to [0,1]^2, then to [0,dstSize]
    Matrix4f s = Matrix4f::scaling
    (
        { dstSize.x / srcSize.x, dstSize.y / srcSize.y, dstSize.z / srcSize.z }
    );

    // translate rectangle to dstOrigin
    Matrix4f t1 = Matrix4f::translation( dstOrigin );

    return t1 * s * t0;
}

// static
Matrix4f Matrix4f::euclidean( const Vector3f& rotationVector, const Vector3f& translation )
{
    Matrix4f output;

    output.setSubmatrix3x3( 0, 0, Matrix3f::rotation( rotationVector.normalized(), rotationVector.norm() ) );
    output.setCol( 3, Vector4f( translation, 1 ) );

    return output;
}

// static
Matrix4f Matrix4f::inverseEuclidean( const Matrix4f& tr )
{
    Matrix4f output;

    Matrix3f rTranspose = tr.getSubmatrix3x3( ).transposed( );
    output.setSubmatrix3x3( 0, 0, rTranspose );

    Vector3f t = tr.getCol( 3 ).xyz;
    Vector3f minusRTransposeT = -rTranspose * t;
    output.setCol( 3, Vector4f( minusRTransposeT, 1 ) );

    return output;
}

// static
Matrix4f Matrix4f::randomRotation( float u0, float u1, float u2 )
{
    return Matrix4f::fromQuat( Quat4f::randomRotation( u0, u1, u2 ) );
}

// static
Matrix4f Matrix4f::lookAt( const Vector3f& eye, const Vector3f& center, const Vector3f& up )
{
    // z is negative forward
    Vector3f z = ( eye - center ).normalized();
    Vector3f y = up;
    Vector3f x = Vector3f::cross( y, z );

    // the x, y, and z vectors define the orthonormal coordinate system
    // the affine part defines the overall translation
    Matrix4f view;

    view.setRow( 0, Vector4f( x, -Vector3f::dot( x, eye ) ) );
    view.setRow( 1, Vector4f( y, -Vector3f::dot( y, eye ) ) );
    view.setRow( 2, Vector4f( z, -Vector3f::dot( z, eye ) ) );
    view.setRow( 3, Vector4f( 0, 0, 0, 1 ) );

    return view;
}

// static
Matrix4f Matrix4f::orthographicProjection( float width, float height, bool directX )
{
    return orthographicProjection( width, height, -1, 1, directX );
}

// static
Matrix4f Matrix4f::orthographicProjection( float width, float height, float zNear, float zFar, bool directX )
{
    Matrix4f m;

    m( 0, 0 ) = 2.0f / width;
    m( 1, 1 ) = 2.0f / height;
    m( 3, 3 ) = 1.0f;

    m( 0, 3 ) = -1;
    m( 1, 3 ) = -1;

    if( directX )
    {
        m( 2, 2 ) = 1.0f / ( zNear - zFar );
        m( 2, 3 ) = zNear / ( zNear - zFar );
    }
    else
    {
        m( 2, 2 ) = 2.0f / ( zNear - zFar );
        m( 2, 3 ) = ( zNear + zFar ) / ( zNear - zFar );
    }

    return m;
}

// static
Matrix4f Matrix4f::orthographicProjection( float left, float right, float bottom, float top, float zNear, float zFar, bool directX )
{
    Matrix4f m;

    m( 0, 0 ) = 2.0f / ( right - left );
    m( 1, 1 ) = 2.0f / ( top - bottom );
    m( 3, 3 ) = 1.0f;

    m( 0, 3 ) = ( left + right ) / ( left - right );
    m( 1, 3 ) = ( top + bottom ) / ( bottom - top );

    if( directX )
    {
        m( 2, 2 ) = 1.0f / ( zNear - zFar );
        m( 2, 3 ) = zNear / ( zNear - zFar );
    }
    else
    {
        m( 2, 2 ) = 2.0f / ( zNear - zFar );
        m( 2, 3 ) = ( zNear + zFar ) / ( zNear - zFar );
    }

    return m;
}

// static
Matrix4f Matrix4f::perspectiveProjection( float left, float right,
    float bottom, float top,
    float zNear, float zFar,
    bool directX )
{
    Matrix4f projection; // zero matrix

    projection( 0, 0 ) = ( 2.0f * zNear ) / ( right - left );
    projection( 1, 1 ) = ( 2.0f * zNear ) / ( top - bottom );
    projection( 0, 2 ) = ( right + left ) / ( right - left );
    projection( 1, 2 ) = ( top + bottom ) / ( top - bottom );
    projection( 3, 2 ) = -1;

    if( directX )
    {
        projection( 2, 2 ) = zFar / ( zNear - zFar );
        projection( 2, 3 ) = ( zNear * zFar ) / ( zNear - zFar );
    }
    else
    {
        projection( 2, 2 ) = ( zNear + zFar ) / ( zNear - zFar );
        projection( 2, 3 ) = ( 2.0f * zNear * zFar ) / ( zNear - zFar );
    }

    return projection;
}

// static
Matrix4f Matrix4f::perspectiveProjection( float fovYRadians, float aspect, float zNear, float zFar, bool directX )
{
    Matrix4f m; // zero matrix

    float yScale = cot( 0.5f * fovYRadians );
    float xScale = yScale / aspect;

    m( 0, 0 ) = xScale;
    m( 1, 1 ) = yScale;
    m( 3, 2 ) = -1;

    if( directX )
    {
        m( 2, 2 ) = zFar / ( zNear - zFar );
        m( 2, 3 ) = zNear * zFar / ( zNear - zFar );
    }
    else
    {
        m( 2, 2 ) = ( zFar + zNear ) / ( zNear - zFar );
        m( 2, 3 ) = 2.f * zFar * zNear / ( zNear - zFar );
    }

    return m;
}

// static
Matrix4f Matrix4f::infinitePerspectiveProjection( float left, float right,
    float bottom, float top,
    float zNear, bool directX )
{
    Matrix4f projection;

    projection( 0, 0 ) = ( 2.0f * zNear ) / ( right - left );
    projection( 1, 1 ) = ( 2.0f * zNear ) / ( top - bottom );
    projection( 0, 2 ) = ( right + left ) / ( right - left );
    projection( 1, 2 ) = ( top + bottom ) / ( top - bottom );
    projection( 3, 2 ) = -1;

    // infinite view frustum
    // just take the limit as far --> inf of the regular frustum
    if( directX )
    {
        projection( 2, 2 ) = -1.0f;
        projection( 2, 3 ) = -zNear;
    }
    else
    {
        projection( 2, 2 ) = -1.0f;
        projection( 2, 3 ) = -2.0f * zNear;
    }

    return projection;
}

// static
Matrix4f Matrix4f::infinitePerspectiveProjection( float fovYRadians, float aspect, float zNear, bool directX )
{
    Matrix4f projection; // zero matrix

    float yScale = cot( 0.5f * fovYRadians );
    float xScale = yScale / aspect;

    projection( 0, 0 ) = xScale;
    projection( 1, 1 ) = yScale;
    projection( 3, 2 ) = -1;

    // infinite view frustum
    // just take the limit as far --> inf of the regular frustum
    if( directX )
    {
        projection( 2, 2 ) = -1.0f;
        projection( 2, 3 ) = -zNear;
    }
    else
    {
        projection( 2, 2 ) = -1.0f;
        projection( 2, 3 ) = -2.0f * zNear;
    }

    return projection;
}

// static
Matrix4f Matrix4f::viewport( float x0, float y0, float width, float height,
    bool directX )
{
    // input NDC x (and resp. y), is in [-1,1]
    // x + 1: [0,2]
    // 0.5 * ( x + 1 ): [0,1]
    // width * 0.5f * ( x + 1 ): [0,width]
    // x0 + width * 0.5f * ( x + 1 ): [x0,x0+width]
    // = x0 + width * 0.5f * x + width * 0.5f
    // = width * 0.5f * x + (x0 + width * 0.5f)

    Matrix4f output;

    output.m00 = 0.5f * width;
    output.m03 = x0 + 0.5f * width;

    output.m11 = 0.5f * height;
    output.m13 = y0 + 0.5f * height;

    // in OpenGL: input NDC z is in [-1,1], rescale to [0,1]
    // z_viewport = 0.5f * z_ndc + 1
    // in DirectX, input NDC z is in [0,1]
    if( directX )
    {
        output.m22 = 1;
    }
    else
    {
        output.m22 = 0.5f;
        output.m23 = 1;
    }

    // in both cases, input NDC w = 1?
    output.m33 = 1;

    return output;
}

// static
Matrix4f Matrix4f::viewport( const Rect2f& rect, bool directX )
{
    return Matrix4f::viewport( rect.origin.x, rect.origin.y, rect.width(), rect.height(), directX );
}

// static
Matrix4f Matrix4f::fromQuat( const Quat4f& q )
{
    Quat4f qq = q.normalized();

    float xx = qq.x * qq.x;
    float yy = qq.y * qq.y;
    float zz = qq.z * qq.z;

    float xy = qq.x * qq.y;
    float zw = qq.z * qq.w;

    float xz = qq.x * qq.z;
    float yw = qq.y * qq.w;

    float yz = qq.y * qq.z;
    float xw = qq.x * qq.w;

    return Matrix4f
        (
        1.0f - 2.0f * ( yy + zz ),      2.0f * ( xy - zw ),             2.0f * ( xz + yw ),             0.0f,
        2.0f * ( xy + zw ),             1.0f - 2.0f * ( xx + zz ),      2.0f * ( yz - xw ),             0.0f,
        2.0f * ( xz - yw ),             2.0f * ( yz + xw ),             1.0f - 2.0f * ( xx + yy ),      0.0f,
        0.0f,                           0.0f,                           0.0f,                           1.0f
        );
}

Matrix4f operator + ( const Matrix4f& x, const Matrix4f& y )
{
    Matrix4f sum;

    for( int k = 0; k < 16; ++k )
    {
        sum[k] = x[k] + y[k];
    }

    return sum;
}

Matrix4f operator - ( const Matrix4f& x, const Matrix4f& y )
{
    Matrix4f difference;

    for( int k = 0; k < 16; ++k )
    {
        difference[k] = x[k] - y[k];
    }

    return difference;
}

Matrix4f operator - ( const Matrix4f& x )
{
    Matrix4f output;

    for( int k = 0; k < 16; ++k )
    {
        output[k] = -x[k];
    }

    return output;
}

Matrix4f operator * ( float f, const Matrix4f& m )
{
    Matrix4f output;

    for( int k = 0; k < 16; ++k )
    {
        output[k] = f * m[k];
    }

    return output;
}

Matrix4f operator * ( const Matrix4f& m, float f )
{
    return f * m;
}

Matrix4f operator / ( const Matrix4f& m, float f )
{
    return ( 1.0f / f ) * m;
}

Vector4f operator * ( const Matrix4f& m, const Vector4f& v )
{
    Vector4f output( 0, 0, 0, 0 );

    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            output[ i ] += m( i, j ) * v[ j ];
        }
    }

    return output;
}

Matrix4f operator * ( const Matrix4f& x, const Matrix4f& y )
{
    Matrix4f product; // zeroes

    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            for( int k = 0; k < 4; ++k )
            {
                product( i, k ) += x( i, j ) * y( j, k );
            }
        }
    }

    return product;
}
