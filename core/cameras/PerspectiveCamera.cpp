#include "libcgt/core/cameras/PerspectiveCamera.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "libcgt/core/cameras/CameraUtils.h"
#include "libcgt/core/math/Arithmetic.h"
#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Quat4f.h"

using libcgt::core::cameras::focalLengthPixelsToFoVRadians;
using libcgt::core::cameras::fovRadiansToFocalLengthPixels;
using libcgt::core::cameras::frustumToIntrinsics;
using libcgt::core::cameras::intrinsicsToFrustum;
using libcgt::core::cameras::GLFrustum;
using libcgt::core::cameras::Intrinsics;
using libcgt::core::math::cubicInterpolate;
using libcgt::core::math::degreesToRadians;
using libcgt::core::math::radiansToDegrees;
using libcgt::core::math::HALF_PI;
using libcgt::core::vecmath::EuclideanTransform;


// static
const PerspectiveCamera PerspectiveCamera::CANONICAL(
    Vector3f( 0, 0, 0 ), Vector3f( 0, 0, -1 ), Vector3f( 0, 1, 0 ),
    HALF_PI, 1.0f, 1.0f, 100.0f );

// static
const PerspectiveCamera PerspectiveCamera::FRONT(
    Vector3f( 0, 0, 5 ), Vector3f( 0, 0, 0 ), Vector3f( 0, 1, 0 ),
    degreesToRadians( 50.0f ), 1.0f, 1.0f, 100.0f );

// static
const PerspectiveCamera PerspectiveCamera::RIGHT(
    Vector3f( 5, 0, 0 ), Vector3f( 0, 0, 0 ), Vector3f( 0, 1, 0 ),
    degreesToRadians( 50.0f ), 1.0f, 1.0f, 100.0f );

// static
const PerspectiveCamera PerspectiveCamera::TOP(
    Vector3f( 0, 5, 0 ), Vector3f( 0, 0, 0 ), Vector3f( 0, 0, -1 ),
    degreesToRadians( 50.0f ), 1.0f, 1.0f, 100.0f );

PerspectiveCamera::PerspectiveCamera(
    const EuclideanTransform& cameraFromWorld,
    const GLFrustum& frustum,
    bool isDirectX )
{
    setCameraFromWorld( cameraFromWorld );
    setFrustum( frustum );
    setDirectX( isDirectX );
}

PerspectiveCamera::PerspectiveCamera(
    const Vector3f& eye, const Vector3f& center, const Vector3f& up,
    const GLFrustum& frustum,
    bool isDirectX )
{
    setLookAt( eye, center, up );
    setFrustum( frustum );
    setDirectX( isDirectX );
}

PerspectiveCamera::PerspectiveCamera(
    const EuclideanTransform& cameraFromWorld,
    const Intrinsics& intrinsics, const Vector2f& imageSize,
    float zNear, float zFar,
    bool isDirectX )
{
    setCameraFromWorld( cameraFromWorld );
    setFrustum( intrinsics, imageSize, zNear, zFar );
    setDirectX( isDirectX );
}

PerspectiveCamera::PerspectiveCamera(
    const Vector3f& eye, const Vector3f& center, const Vector3f& up,
    float fovYRadians, float aspect,
    float zNear, float zFar,
    bool isDirectX )
{
    setLookAt( eye, center, up );
    setFrustum( GLFrustum::makeSymmetricPerspective(
        fovYRadians, aspect, zNear, zFar ) );
    setDirectX( isDirectX );
}

void PerspectiveCamera::setFrustum( const Intrinsics& intrinsics,
    const Vector2f& imageSize )
{
    setFrustum( intrinsicsToFrustum( intrinsics, imageSize,
        frustum().zNear, frustum().zFar ) );
}

void PerspectiveCamera::setFrustum( const Intrinsics& intrinsics,
    const Vector2f& imageSize, float zNear, float zFar )
{
    assert( zNear > 0 );
    setFrustum( intrinsicsToFrustum( intrinsics, imageSize, zNear, zFar ) );
}

Intrinsics PerspectiveCamera::intrinsics( const Vector2f& screenSize ) const
{
    return frustumToIntrinsics( frustum(), screenSize );
}

// virtual
Matrix4f PerspectiveCamera::projectionMatrix() const
{
    if( isinf( frustum().zFar ) )
    {
        return Matrix4f::infinitePerspectiveProjection(
            frustum().left, frustum().right,
            frustum().bottom, frustum().top,
            frustum().zNear, isDirectX() );
    }
    else
    {
        return Matrix4f::perspectiveProjection(
            frustum().left, frustum().right,
            frustum().bottom, frustum().top,
            frustum().zNear, frustum().zFar, isDirectX() );
    }
}

Matrix4f PerspectiveCamera::jitteredProjectionMatrix( float eyeX, float eyeY,
    float focusZ ) const
{
    float dx = -eyeX * frustum().zNear / focusZ;
    float dy = -eyeY * frustum().zNear / focusZ;

    if( isinf( frustum().zFar ) )
    {
        return Matrix4f::infinitePerspectiveProjection(
            frustum().left + dx, frustum().right + dx,
            frustum().bottom + dy, frustum().top + dy,
            frustum().zNear, isDirectX() );
    }
    else
    {
        return Matrix4f::perspectiveProjection(
            frustum().left + dx, frustum().right + dx,
            frustum().bottom + dy, frustum().top + dy,
            frustum().zNear, frustum().zFar, isDirectX() );
    }
}

Matrix4f PerspectiveCamera::jitteredViewProjectionMatrix(
    float eyeX, float eyeY, float focusZ ) const
{
    return
    (
        jitteredProjectionMatrix( eyeX, eyeY, focusZ ) *
        jitteredViewMatrix( eyeX, eyeY )
    );
}

// virtual
std::string PerspectiveCamera::toString() const
{
    std::ostringstream sstream;

    sstream << "perspective_camera" << "\n";
    sstream << "eye " << eye().toString() << "\n";
    sstream << "right " << right().toString() << "\n";
    sstream << "up " << up().toString() << "\n";
    sstream << "back " << back().toString() << "\n";
    sstream << "frustum_left " << frustum().left << "\n";
    sstream << "frustum_right " << frustum().right << "\n";
    sstream << "frustum_bottom " << frustum().bottom << "\n";
    sstream << "frustum_top " << frustum().top << "\n";
    sstream << "frustum_zNear " << frustum().zNear << "\n";
    sstream << "frustum_zFar " << frustum().zFar << "\n";

    return sstream.str();
}

std::vector< Vector4f > PerspectiveCamera::frustumLines() const
{
    std::vector< Vector3f > corners = frustumCorners();
    std::vector< Vector4f > output( 24 );

    Vector3f e = eye();

    // 4 lines from eye to each far corner
    output[ 0] = Vector4f( e, 1 );
    output[ 1] = Vector4f( corners[4], 1 );

    output[ 2] = Vector4f( e, 1 );
    output[ 3] = Vector4f( corners[5], 1 );

    output[ 4] = Vector4f( e, 1 );
    output[ 5] = Vector4f( corners[6], 1 );

    output[ 6] = Vector4f( e, 1 );
    output[ 7] = Vector4f( corners[7], 1 );

    // 4 lines between near corners
    output[ 8] = Vector4f( corners[0], 1 );
    output[ 9] = Vector4f( corners[1], 1 );

    output[10] = Vector4f( corners[1], 1 );
    output[11] = Vector4f( corners[2], 1 );

    output[12] = Vector4f( corners[2], 1 );
    output[13] = Vector4f( corners[3], 1 );

    output[14] = Vector4f( corners[3], 1 );
    output[15] = Vector4f( corners[0], 1 );

    // 4 lines between far corners
    output[16] = Vector4f( corners[4], 1 );
    output[17] = Vector4f( corners[5], 1 );

    output[18] = Vector4f( corners[5], 1 );
    output[19] = Vector4f( corners[6], 1 );

    output[20] = Vector4f( corners[6], 1 );
    output[21] = Vector4f( corners[7], 1 );

    output[22] = Vector4f( corners[7], 1 );
    output[23] = Vector4f( corners[4], 1 );

    // TODO: handle infinite z plane

    return output;
}

// static
PerspectiveCamera PerspectiveCamera::lerp( const PerspectiveCamera& c0,
    const PerspectiveCamera& c1, float t )
{
    bool isDirectX = c0.isDirectX();

    GLFrustum f = GLFrustum::lerp( c0.frustum(), c1.frustum(), t);

    // TODO(jiawen): lerp between EuclideanTransforms?
    Vector3f eye = libcgt::core::math::lerp( c0.eye(), c1.eye(), t );

    Quat4f q0 = Quat4f::fromRotatedBasis(
        c0.right(), c0.up(), -( c0.forward() ) );
    Quat4f q1 = Quat4f::fromRotatedBasis(
        c1.right(), c1.up(), -( c1.forward() ) );
    Quat4f q = Quat4f::slerp( q0, q1, t );

    Vector3f x = q.rotateVector( Vector3f::RIGHT );
    Vector3f y = q.rotateVector( Vector3f::UP );
    Vector3f z = q.rotateVector( -Vector3f::FORWARD );

    Vector3f center = eye - z;

    return PerspectiveCamera
    (
        eye, center, y,
        f,
        c0.isDirectX()
    );
}

// static
PerspectiveCamera PerspectiveCamera::cubicInterpolate(
    const PerspectiveCamera& c0,
    const PerspectiveCamera& c1,
    const PerspectiveCamera& c2,
    const PerspectiveCamera& c3,
    float t )
{
    GLFrustum f = GLFrustum::lerp(c0.frustum(), c1.frustum(), t);

    Vector3f eye = ::cubicInterpolate(
        c0.eye(), c1.eye(), c2.eye(), c3.eye(), t );

    Quat4f q0 = Quat4f::fromRotatedBasis(
        c0.right(), c0.up(), -( c0.forward() ) );
    Quat4f q1 = Quat4f::fromRotatedBasis(
        c1.right(), c1.up(), -( c1.forward() ) );
    Quat4f q2 = Quat4f::fromRotatedBasis(
        c2.right(), c2.up(), -( c2.forward() ) );
    Quat4f q3 = Quat4f::fromRotatedBasis(
        c3.right(), c3.up(), -( c3.forward() ) );

    Quat4f q = Quat4f::cubicInterpolate( q0, q1, q2, q3, t );

    Vector3f x = q.rotateVector( Vector3f::RIGHT );
    Vector3f y = q.rotateVector( Vector3f::UP );
    Vector3f z = q.rotateVector( -Vector3f::FORWARD );

    Vector3f center = eye - z;

    return PerspectiveCamera
    (
        eye, center, y,
        f,
        c0.isDirectX()
    );
}

bool operator == ( const PerspectiveCamera& c0, const PerspectiveCamera& c1 )
{
    const Camera& cc0 = c0;
    const Camera& cc1 = c1;
    return cc0 == cc1;
}

bool operator != ( const PerspectiveCamera& c0, const PerspectiveCamera& c1 )
{
    return !( c0 == c1 );
}
