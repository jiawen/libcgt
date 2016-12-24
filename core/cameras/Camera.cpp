#include "libcgt/core/cameras/Camera.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#include "libcgt/core/cameras/CameraUtils.h"
#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Vector4f.h"
#include "libcgt/core/vecmath/Quat4f.h"

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
        // Vertices go around in order counterclockwise.
        // DirectX uses NDC z in [0,1]
        cubePoint[0] = ( ( i & 1 ) ^ ( ( i & 2 ) >> 1 ) ? 1.f : -1.f );
        cubePoint[1] = ( i & 2 ) ? 1.f : -1.f;
        cubePoint[2] = ( i & 4 ) ? 1.f : ( m_directX ? 0.f : -1.f );

        // This would be the hypercube ordering.
        // cubePoint[0] = ( i & 1 ) ? 1.f : -1.f;
        // cubePoint[1] = ( i & 2 ) ? 1.f : -1.f;
        // DirectX uses NDC z in [0,1]
        // cubePoint[2] = ( i & 4 ) ? 1.f : ( m_directX ? 0.f : -1.f );

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

void Camera::setCameraFromWorld( const EuclideanTransform& cfw )
{
    m_cameraFromWorld = cfw;
}

EuclideanTransform Camera::worldFromCamera() const
{
    return inverse( m_cameraFromWorld );
}

void Camera::setWorldFromCamera( const EuclideanTransform& wfc )
{
    m_cameraFromWorld = inverse( wfc );
}

void Camera::setCameraFromWorldMatrix( const Matrix4f& cfw )
{
    m_cameraFromWorld = EuclideanTransform::fromMatrix( cfw );
}

void Camera::setWorldFromCameraMatrix( const Matrix4f& wfc )
{
    m_cameraFromWorld = inverse( EuclideanTransform::fromMatrix( wfc ) );
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
    return m_cameraFromWorld.asMatrix();
}

Matrix4f Camera::inverseViewMatrix() const
{
    return inverse( m_cameraFromWorld ).asMatrix();
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

Vector3f Camera::eyeFromWorld( const Vector3f& world ) const
{
    return transformPoint( m_cameraFromWorld, world );
}

Vector4f Camera::eyeFromWorld( const Vector4f& world ) const
{
    return viewMatrix() * world;
}

Vector4f Camera::clipFromEye( const Vector4f& eye ) const
{
    return projectionMatrix() * eye;
}

Vector4f Camera::ndcFromClip( const Vector4f& clip ) const
{
    Vector4f ndc = clip.homogenized();
    ndc.w = clip.w;
    return ndc;
}

Vector4f Camera::screenFromNDC( const Vector4f& ndc,
    const Vector2f& screenSize ) const
{
    // In Direct3D, ndc.z is in [0, 1], so leave it as is.
    // In OpenGL, ndc.z is in [-1, 1], so shift it.
    return Vector4f
    (
        screenSize.x * 0.5f * ( ndc.x + 1.0f ),
        screenSize.y * 0.5f * ( ndc.y + 1.0f ),
        m_directX ? ndc.z : 0.5f * ( ndc.z + 1.0f ),
        ndc.w
    );
}

Vector4f Camera::screenFromClip( const Vector4f& clip,
    const Vector2f& screenSize ) const
{
    Vector4f ndc = ndcFromClip( clip );
    return screenFromNDC( ndc, screenSize );
}

Vector4f Camera::screenFromEye( const Vector4f& eye,
    const Vector2f& screenSize ) const
{
    Vector4f clip = clipFromEye( eye );
    return screenFromClip( clip, screenSize );
}

Vector4f Camera::screenFromWorld( const Vector4f& world,
    const Vector2f& screenSize ) const
{
    Vector4f eye = eyeFromWorld( world );
    return screenFromEye( eye, screenSize );
}

// virtual
Vector4f Camera::eyeFromScreen( const Vector2i& xy,
    float depth, const Vector2f& screenSize ) const
{
    return eyeFromScreen( xy + 0.5f, depth, screenSize );
}

// virtual
Vector4f Camera::eyeFromScreen( const Vector2f& xy,
    float depth, const Vector2f& screenSize ) const
{
    Vector2f ndcXY = ndcFromScreen( xy, screenSize );

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

Vector4f Camera::worldFromScreen( const Vector2i& xy, float depth,
    const Vector2f& screenSize ) const
{
    return worldFromScreen( xy + 0.5f, depth, screenSize );
}

Vector4f Camera::worldFromScreen( const Vector2f& xy, float depth,
    const Vector2f& screenSize ) const
{
    Vector4f eye = eyeFromScreen( xy, depth, screenSize );
    return inverseViewMatrix() * eye;
}

Vector3f Camera::eyeDirectionFromScreen( const Vector2f& xy,
    const Vector2f& screenSize ) const
{
    // TODO: can use Intrinsics functions instead, which is less math.
    Vector4f eyePoint = eyeFromScreen( xy, zNear(), screenSize );
    return eyePoint.xyz.normalized();
}

Vector3f Camera::eyeDirectionFromScreen( const Vector2i& xy,
    const Vector2f& screenSize ) const
{
    return eyeDirectionFromScreen( xy + 0.5f, screenSize );
}

Vector3f Camera::worldDirectionFromScreen( const Vector2i& xy,
    const Vector2f& screenSize ) const
{
    return worldDirectionFromScreen( xy + 0.5f, screenSize );
}

Vector3f Camera::worldDirectionFromScreen( const Vector2f& xy,
    const Vector2f& screenSize ) const
{
    Vector4f worldPoint = worldFromScreen( xy, zNear(), screenSize );
    return ( worldPoint.xyz - eye() ).normalized();
}

// static
Vector2f Camera::ndcFromScreen( const Vector2f& xy,
    const Vector2f& screenSize )
{
    return 2 * xy / screenSize - Vector2f{ 1 };
}

// static
Vector2f Camera::ndcFromScreen( const Vector2i& xy,
    const Vector2f& screenSize )
{
    return ndcFromScreen( xy + 0.5f, screenSize );
}

// static
void Camera::copyFrustum( const Camera& from, Camera& to )
{
    to.m_frustum = from.m_frustum;
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

bool operator == ( const Camera& c0, const Camera& c1 )
{
    return( c0.cameraFromWorld() == c1.cameraFromWorld() &&
        c0.frustum() == c1.frustum() &&
        c0.isDirectX() == c1.isDirectX() );
}

bool operator != ( const Camera& c0, const Camera& c1 )
{
    return !( c0 == c1 );
}
