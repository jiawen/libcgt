#include "Camera.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#include "CameraUtils.h"
#include "math/MathUtils.h"
#include "vecmath/Vector4f.h"
#include "vecmath/Quat4f.h"

using libcgt::core::vecmath::EuclideanTransform;
using libcgt::core::cameras::frustumToIntrinsics;
using libcgt::core::cameras::GLFrustum;
using libcgt::core::cameras::Intrinsics;
using libcgt::core::vecmath::inverse;
using libcgt::core::vecmath::transformPoint;
using libcgt::core::vecmath::transformVector;

bool Camera::isDirectX() const
{
    return m_directX;
}

const GLFrustum& Camera::frustum() const
{
    return m_frustum;
}

float Camera::aspectRatio() const
{
    return m_frustum.aspectRatio();
}

float Camera::zNear() const
{
    return m_frustum.zNear;
}

float Camera::zFar() const
{
    return m_frustum.zFar;
}

bool Camera::isZFarInfinite() const
{
    return isinf( m_frustum.zFar );
}

Matrix4f Camera::inverseProjectionMatrix() const
{
    return projectionMatrix().inverse();
}

std::vector< Vector3f > Camera::frustumCorners() const
{
    std::vector< Vector3f > corners( 8 );

    Vector4f cubePoint( 0.f, 0.f, 0.f, 1.f );
    Matrix4f invViewProj = inverseViewProjectionMatrix();

    // take the NDC cube and unproject it
    for( int i = 0; i < 8; ++i )
    {
        // so vertices go around in order counterclockwise
        cubePoint[0] = ( ( i & 1 ) ^ ( ( i & 2 ) >> 1 ) ? 1.f : -1.f );
        cubePoint[1] = ( i & 2 ) ? 1.f : -1.f;
        cubePoint[2] = ( i & 4 ) ? 1.f : ( m_directX ? 0.f : -1.f ); // DirectX uses NDC z in [0,1]

        // This would be the hypercube ordering.
        // cubePoint[0] = ( i & 1 ) ? 1.f : -1.f;
        // cubePoint[1] = ( i & 2 ) ? 1.f : -1.f;
        // cubePoint[2] = ( i & 4 ) ? 1.f : ( m_directX ? 0.f : -1.f ); // DirectX uses NDC z in [0,1]

        corners[ i ] = ( invViewProj * cubePoint ).homogenized().xyz;
    }

    return corners;
}

std::vector< Plane3f > Camera::frustumPlanes() const
{
    auto corners = frustumCorners();
    std::vector< Plane3f > planes( 6 );

    // left
    planes[ 0 ] = Plane3f( corners[ 0 ], corners[ 3 ], corners[ 4 ] );
    // bottom
    planes[ 1 ] = Plane3f( corners[ 1 ], corners[ 0 ], corners[ 4 ] );
    // right
    planes[ 2 ] = Plane3f( corners[ 2 ], corners[ 1 ], corners[ 5 ] );
    // top
    planes[ 3 ] = Plane3f( corners[ 3 ], corners[ 2 ], corners[ 6 ] );
    // near
    planes[ 4 ] = Plane3f( corners[ 0 ], corners[ 1 ], corners[ 2 ] );
    // far
    planes[ 5 ] = Plane3f( corners[ 5 ], corners[ 4 ], corners[ 6 ] );

    return planes;
}

EuclideanTransform Camera::cameraFromWorld() const
{
    return m_cameraFromWorld;
}

void Camera::setCameraFromWorld( const libcgt::core::vecmath::EuclideanTransform& cfw )
{
    m_cameraFromWorld = cfw;
}

EuclideanTransform Camera::worldFromCamera() const
{
    return inverse( m_cameraFromWorld );
}

void Camera::setWorldFromCamera( const libcgt::core::vecmath::EuclideanTransform& wfc )
{
    m_cameraFromWorld = inverse( wfc );
}

void Camera::setCameraFromWorldMatrix( const Matrix4f& cfw )
{
    cfw.decomposeRotationTranslation( m_cameraFromWorld.rotation,
        m_cameraFromWorld.translation );
}

void Camera::setWorldFromCameraMatrix( const Matrix4f& wfc )
{
    Matrix4f::inverseEuclidean( wfc ).decomposeRotationTranslation(
        m_cameraFromWorld.rotation, m_cameraFromWorld.translation);
}

void Camera::setLookAt( const Vector3f& eye, const Vector3f& center,
    const Vector3f& up )
{
    setCameraFromWorldMatrix( Matrix4f::lookAt( eye, center, up ) );
}

Vector3f Camera::eye() const
{
    return transformPoint( worldFromCamera(), { 0, 0, 0 } );
}

void Camera::setEye( const Vector3f& eye )
{
    m_cameraFromWorld.translation = -eye;
}

Vector3f Camera::right() const
{
    return transformVector( worldFromCamera(), Vector3f::RIGHT );
}

Vector3f Camera::up() const
{
    return transformVector( worldFromCamera(), Vector3f::UP );
}

Vector3f Camera::back() const
{
    return transformVector( worldFromCamera(), -Vector3f::FORWARD );
}

Vector3f Camera::forward() const
{
    return transformVector( worldFromCamera(), Vector3f::FORWARD );
}

Matrix4f Camera::viewMatrix() const
{
    return m_cameraFromWorld;
}

Matrix4f Camera::inverseViewMatrix() const
{
    return viewMatrix().inverse();
}

Matrix4f Camera::jitteredViewMatrix( float eyeX, float eyeY ) const
{
    Vector3f jitteredEye = eye() + eyeX * right() + eyeY * up();
    return Matrix4f::lookAt( jitteredEye, eye() + forward(), up() );
}

Matrix4f Camera::viewProjectionMatrix() const
{
    return projectionMatrix() * viewMatrix();
}

Matrix4f Camera::inverseViewProjectionMatrix() const
{
    return viewProjectionMatrix().inverse();
}

Intrinsics Camera::intrinsics( const Vector2f& screenSize ) const
{
    return frustumToIntrinsics( m_frustum, screenSize );
}

Vector4f Camera::worldToEye( const Vector4f& world ) const
{
    return viewMatrix() * world;
}

Vector4f Camera::eyeToClip( const Vector4f& eye ) const
{
    return projectionMatrix() * eye;
}

Vector4f Camera::clipToNDC( const Vector4f& clip ) const
{
    Vector4f ndc = clip.homogenized();
    ndc.w = clip.w;
    return ndc;
}

Vector4f Camera::eyeToScreen( const Vector4f& eye, const Vector2f& screenSize ) const
{
    Vector4f clip = eyeToClip( eye );
    Vector4f ndc = clip.homogenized();
    ndc.w = clip.w;
    return ndcToScreen( ndc, screenSize );
}

Vector4f Camera::ndcToScreen( const Vector4f& ndc, const Vector2f& screenSize ) const
{
    return Vector4f
    (
        screenSize.x * 0.5f * ( ndc.x + 1.0f ),
        screenSize.y * 0.5f * ( ndc.y + 1.0f ),
        m_directX ? ndc.z : 0.5f * ( ndc.z + 1.0f ),
        ndc.w
    );
}

Vector4f Camera::worldToScreen( const Vector4f& world, const Vector2f& screenSize ) const
{
    Vector4f eye = worldToEye( world );
    return eyeToScreen( eye, screenSize );
}


// virtual
Vector4f Camera::screenToEye( const Vector2i& xy, float depth, const Vector2f& screenSize ) const
{
    return screenToEye( Vector2f{ xy.x + 0.5f, xy.y + 0.5f }, depth, screenSize );
}

// virtual
Vector4f Camera::screenToEye( const Vector2f& xy, float depth, const Vector2f& screenSize ) const
{
    Vector2f ndcXY = screenToNDC( xy, screenSize );

    // forward transformation:
    //
    // depth = -zEye
    //
    // xClip = xEye * ( 2 * zNear ) / ( right - left ) + zEye * ( right + left ) / ( right - left )
    // wClip = -zEye
    //
    // xNDC = xClip / wClip = xClip / -zEye = xClip / depth
    //
    // -->
    // inverse transformation:
    //
    // xClip = xNDC * depth
    //
    // xClip - zEye * ( right + left ) / ( right - left ) = xEye * ( 2 * zNear ) / ( right - left )
    // xClip + depth * ( right + left ) / ( right - left ) = xEye * ( 2 * zNear ) / ( right - left )
    //
    // xEye = [ xClip + depth * ( right + left ) / ( right - left ) ] / [ ( 2 * zNear ) / ( right - left ) ]

    float xClip = ndcXY.x * depth;
    float yClip = ndcXY.y * depth;

    float xNumerator = xClip + depth * ( m_frustum.right + m_frustum.left ) / ( m_frustum.right - m_frustum.left );
    float yNumerator = yClip + depth * ( m_frustum.top + m_frustum.bottom ) / ( m_frustum.top - m_frustum.bottom );

    float xDenominator = ( 2 * m_frustum.zNear ) / ( m_frustum.right - m_frustum.left );
    float yDenominator = ( 2 * m_frustum.zNear ) / ( m_frustum.top - m_frustum.bottom );

    float xEye = xNumerator / xDenominator;
    float yEye = yNumerator / yDenominator;
    float zEye = -depth;

    return Vector4f( xEye, yEye, zEye, 1 );
}

Vector4f Camera::screenToWorld( const Vector2i& xy, float depth, const Vector2f& screenSize ) const
{
    return screenToWorld( Vector2f{ xy.x + 0.5f, xy.y + 0.5f }, depth, screenSize );
}

Vector4f Camera::screenToWorld( const Vector2f& xy, float depth, const Vector2f& screenSize ) const
{
    Vector4f eye = screenToEye( xy, depth, screenSize );
    return inverseViewMatrix() * eye;
}

Vector3f Camera::screenToDirection( const Vector2i& xy, const Vector2f& screenSize ) const
{
    return screenToDirection
    (
        Vector2f{ xy.x + 0.5f, xy.y + 0.5f },
        Rect2f( screenSize )
    );
}

Vector3f Camera::screenToDirection( const Vector2f& xy, const Vector2f& screenSize ) const
{
    return screenToDirection
    (
        xy,
        Rect2f( screenSize )
    );
}

Vector3f Camera::screenToDirection( const Vector2f& xy, const Rect2f& viewport ) const
{
    // Convert from screen coordinates to NDC.
    float ndcX = 2 * ( xy.x - viewport.origin().x ) / viewport.width() - 1;
    float ndcY = 2 * ( xy.y - viewport.origin().y ) / viewport.height() - 1;

    Vector4f clipPoint( ndcX, ndcY, 0, 1 );
    Vector4f eyePoint = inverseProjectionMatrix() * clipPoint;
    Vector4f worldPoint = inverseViewMatrix() * eyePoint;

    Vector3f pointOnNearPlane = worldPoint.homogenized().xyz;

    // TODO: can use pixelToWorld on z = zNear(), but pixelToWorld needs a viewport version

    return ( pointOnNearPlane - eye() ).normalized();
}

// static
Vector2f Camera::screenToNDC( const Vector2f& xy, const Vector2f& screenSize )
{
    // convert from screen coordinates to NDC
    float ndcX = 2 * xy.x / screenSize.x - 1;
    float ndcY = 2 * xy.y / screenSize.y - 1;

    return{ ndcX, ndcY };
}

// static
void Camera::copyPose( const Camera& from, Camera& to )
{
    to.m_cameraFromWorld = from.m_cameraFromWorld;
}

void Camera::setFrustum( const GLFrustum& frustum )
{
    assert( frustum.zNear > 0 );
    assert( frustum.zFar > frustum.zNear );
    assert( frustum.left < frustum.right );
    assert( frustum.bottom < frustum.top );

    m_frustum = frustum;
}

void Camera::setDirectX( bool directX )
{
    m_directX = directX;
}
