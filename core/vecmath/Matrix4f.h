#pragma once

#include <string>
#include "libcgt/core/vecmath/Vector4f.h"

class Matrix2f;
class Matrix3f;
class Quat4f;
class Vector3f;
class Rect2f;

// 4x4 matrix, stored in column major order (FORTRAN / OpenGL style)
class Matrix4f
{
public:

    // Rotate about the X axis by 180 degrees.
    static Matrix4f ROTATE_X_180;
    static Matrix4f ROTATE_Y_180;
    static Matrix4f ROTATE_Z_180;

    // Default is the zero matrix.
    Matrix4f();

    explicit Matrix4f( float fill );

    // Construct 4x4 matrix directly from elements.
    // Elements are written in row-major order for human-readability
    // (and stored column major, as usual).
    Matrix4f( float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33 );

    // setColumns = true ==> sets the columns of the matrix to be [v0 v1 v2 v3]
    // otherwise, sets the rows
    Matrix4f( const Vector4f& v0, const Vector4f& v1,
        const Vector4f& v2, const Vector4f& v3,
        bool setColumns = true );

    Matrix4f( const Matrix4f& m ) = default;
    Matrix4f& operator = ( const Matrix4f& m ) = default;

    // read / write element (i,j)
    const float& operator () ( int i, int j ) const;
    float& operator () ( int i, int j );

    // get/set row i
    Vector4f getRow( int i ) const;
    void setRow( int i, const Vector4f& v );

    // get/set column j
    Vector4f getCol( int j ) const;
    void setCol( int j, const Vector4f& v );

    // gets/sets the 2x2 submatrix
    // starting with upper left corner at (i0, j0)
    Matrix2f getSubmatrix2x2( int i0 = 0, int j0 = 0 ) const;
    void setSubmatrix2x2( int i0, int j0, const Matrix2f& m );

    // gets/sets the 3x3 submatrix
    // starting with upper left corner at (i0, j0)
    Matrix3f getSubmatrix3x3( int i0 = 0, int j0 = 0 ) const;
    void setSubmatrix3x3( int i0, int j0, const Matrix3f& m );

    float determinant() const;
    Matrix4f inverse() const; // TODO: invert(), inverted()?
    Matrix4f inverse( bool& isSingular, float epsilon = 0.f ) const; // TODO: invert(), inverted()?

    void transpose();
    Matrix4f transposed() const;

    // ----- Decompositions -----
    // inverse transpose of top left 3x3 submatrix
    Matrix3f normalMatrix() const;

    // inverse transpose of top left 3x3 submatrix on top left, 0 elsewhere
    Matrix4f normalMatrix4x4() const;

    // ---- Utility ----
    // implicit cast to pointer
    operator const float* () const;
    operator float* ();

    std::string toString() const;

    // Uses this to transform a point v.
    // Appends a homogeneous coordinate 1, transforms, then extracts xy.
    Vector3f transformPoint( const Vector3f& p ) const;

    // Uses this to transform a vector v.
    // Appends a homogeneous coordinate 0, transforms, then extracts xy.
    Vector3f transformVector( const Vector3f& v ) const;

    // uses this to transform a normal vector n
    // equivalent to normalMatrix() * n
    Vector3f transformNormal( const Vector3f& n ) const;

    // ---- Common graphics matrices ----
    static Matrix4f ones();
    static Matrix4f identity();
    static Matrix4f rotateX( float radians );
    static Matrix4f rotateY( float radians );
    static Matrix4f rotateZ( float radians );
    static Matrix4f rotation( const Vector3f& axis, float radians );
    static Matrix4f rotation( const Vector3f& axisAngle );
    static Matrix4f scaling( const Vector3f& xyz );
    static Matrix4f uniformScaling( float s );
    static Matrix4f translation( const Vector3f& xyz );
    static Matrix4f scaleTranslate(
        const Vector3f& srcOrigin, const Vector3f& srcSize,
        const Vector3f& dstOrigin, const Vector3f& dstSize );

    // Construct a Euclidean transformation matrix from a rotation vector and a
    // translation.
    // E = [R|t]. Rotation is applied first, then translation.
    // The rotation vector's direction is the axis, its norm is the number of
    // radians, counterclockwise.
    static Matrix4f euclidean( const Vector3f& rotationVector,
        const Vector3f& translation );

    // Construct a Similarity transformation matrix from a uniform scale, a
    // rotation vector and a translation.
    // S = [sR|t]. Rotation is applied first, then scale, then translation.
    // The rotation vector's direction is the axis, its norm is the number of
    // radians, counterclockwise.
    static Matrix4f similarity( const Vector3f& rotationVector,
        float scale, const Vector3f& translation );

    // Invert a Euclidean transformation.
    // If E is a Euclidean transformation: E = T*R = [R|t]
    // (apply R, then T), computes inv(E):
    // inv(T*R) = inv(R)*inv(T)
    //          = transpose(R) * (-T)    (apply -T, then transpose(R))
    //          = [R'|-R'*t]
    static Matrix4f inverseEuclidean( const Matrix4f& e );

    // Invert a Euclidean transformation.
    // If S is a Euclidean transformation: S = T*s*R = [sR|t]
    // (apply R, then s, then T), computes inv(S):
    // inv(T*s*R) = inv(R)*inv(s)*inv(T)
    //          = transpose(R) * (1/s) * (-T)    (apply -T, then 1/s, then R')
    //          = similarity(
    static Matrix4f inverseSimilarity( const Matrix4f& s );

    static Matrix4f lookAt( const Vector3f& eye, const Vector3f& center, const Vector3f& up );

    // orthographicProjection from (0,0) to (width,height), zNear = -1, zFar = 1
    static Matrix4f orthographicProjection( float width, float height, bool directX );
    // orthographicProjection from (0,0) to (width,height)
    static Matrix4f orthographicProjection( float width, float height, float zNear, float zFar, bool directX );
    // orthographicProjection from (left, bottom) to (right, top)
    static Matrix4f orthographicProjection( float left, float right, float bottom, float top, float zNear, float zFar, bool directX );

    // [potentially] skewed perspective projection matrix defined by 4 points on the near plane, zNear and zFar
    static Matrix4f perspectiveProjection( float left, float right, float bottom, float top, float zNear, float zFar, bool directX );
    // non-skewed perspective projection matrix defined by a field of view, aspect ratio, zNear, and zFar
    static Matrix4f perspectiveProjection( float fovYRadians, float aspect, float zNear, float zFar, bool directX );

    // [potentially] skewed perspective projection matrix defined by 4 points on the near plane, zNear, and zFar at positive infinity
    static Matrix4f infinitePerspectiveProjection( float left, float right, float bottom, float top, float zNear, bool directX );
    // non-skewed perspective projection matrix defined by a field of view, aspect ratio, zNear, and zFar at positive infinity
    static Matrix4f infinitePerspectiveProjection( float fovYRadians, float aspect, float zNear, bool directX );

    // Same as viewport( 0, 0, width, height, directX )
    static Matrix4f viewport( float width, float height, bool directX );

    // Constructs the matrix mapping NDC coordinates
    // (OpenGL: [-1,1]^3, Direct3D: [-1,1]^2 x [0,1])
    // to window coordinates: [x0, x0 + width), [y0, y0 + height), [0,1]
    // TODO: OpenGL has glDepthRange, (default to [0,1]) for window coordinates' z as well
    static Matrix4f viewport( float x0, float y0, float width, float height, bool directX );
    static Matrix4f viewport( const Rect2f& rect, bool directX );

    // Construct a rotation matrix represented by a unit quaternion (does not
    // normalize q).
    static Matrix4f fromQuat( const Quat4f& q );

    // returns an orthogonal matrix that's a uniformly distributed rotation
    // given u[i] is a uniformly distributed random number in [0,1]
    static Matrix4f randomRotation( float u0, float u1, float u2 );

    union
    {
        struct
        {
            float m00;
            float m10;
            float m20;
            float m30;

            float m01;
            float m11;
            float m21;
            float m31;

            float m02;
            float m12;
            float m22;
            float m32;

            float m03;
            float m13;
            float m23;
            float m33;
        };
        struct
        {
            Vector4f column0;
            Vector4f column1;
            Vector4f column2;
            Vector4f column3;
        };
        float m_elements[ 16 ];
    };

};

Matrix4f operator + ( const Matrix4f& x, const Matrix4f& y );
Matrix4f operator - ( const Matrix4f& x, const Matrix4f& y );
// Negate.
Matrix4f operator - ( const Matrix4f& x );

// Multiply matrix by scalar.
Matrix4f operator * ( float f, const Matrix4f& m );
Matrix4f operator * ( const Matrix4f& m, float f );

// Divide matrix by scalar.
Matrix4f operator / ( const Matrix4f& m, float f );

// Matrix-Vector multiplication.
// 4x4 * 4x1 ==> 4x1
Vector4f operator * ( const Matrix4f& m, const Vector4f& v );

// Matrix-Matrix multiplication.
Matrix4f operator * ( const Matrix4f& x, const Matrix4f& y );
