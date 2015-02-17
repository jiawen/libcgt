#include "cameras/Camera.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#include "cameras/CameraUtils.h"
#include "math/MathUtils.h"
#include "vecmath/Vector4f.h"
#include "vecmath/Quat4f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

void Camera::setDirectX( bool directX )
{
	m_directX = directX;
}

void Camera::getFrustum( float& left, float& right,
	float& bottom, float& top,
	float& zNear, float& zFar ) const
{
    left = m_left;
    right = m_right;

    bottom = m_bottom;
    top = m_top;

    zNear = m_zNear;
    zFar = m_zFar;
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

		// this would be the hypercube ordering
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

bool Camera::isZFarInfinite() const
{
	return isinf( m_zFar );
}

void Camera::setFrustum( float left, float right,
	float bottom, float top,
	float zNear, float zFar )
{
	assert( zNear > 0 );
    assert( zFar > zNear );
    assert( left < right );
    assert( bottom < top );

	m_left = left;
	m_right = right;

	m_bottom = bottom;
	m_top = top;

	m_zNear = zNear;
	m_zFar = zFar;
}

void Camera::setFrustumFromIntrinsics( const Vector2f& focalLengthPixels, const Vector2f& principalPointPixels,
	const Vector2f& imageSize )
{
	CameraUtils::intrinsicsToFrustum
	(
		focalLengthPixels, principalPointPixels,
		imageSize,

		m_zNear,
		m_left, m_right,
		m_bottom, m_top
	);
}

void Camera::setFrustumFromIntrinsics( const Vector2f& focalLengthPixels, const Vector2f& principalPointPixels,
	const Vector2f& imageSize,
	float zNear, float zFar )
{
	assert( zNear > 0 );	

	m_zNear = zNear;
	m_zFar = zFar;

	setFrustumFromIntrinsics( focalLengthPixels, principalPointPixels, imageSize );
}

void Camera::getLookAt( Vector3f& eye,
	Vector3f& center,
	Vector3f& up ) const
{
	eye = m_eye;
	center = m_center;
	up = m_up;
}

void Camera::setLookAt( const Vector3f& eye,
	const Vector3f& center,
	const Vector3f& up )
{
	m_eye = eye;
	m_center = center;
	m_up = up;
}

void Camera::setLookAtFromInverseViewMatrix( const Matrix4f& ivm )
{
	float d = ( center() - eye() ).norm();
	setLookAt
	(
		ivm.transformPoint( Vector3f::ZERO ),
		ivm.transformPoint( Vector3f( 0, 0, -d ) ),
		ivm.transformVector( Vector3f::UP )
	);
}

Vector3f Camera::eye() const
{
	return m_eye;
}

void Camera::setEye( const Vector3f& eye )
{
	m_eye = eye;
}

Vector3f Camera::center() const
{
	return m_center;
}

void Camera::setCenter( const Vector3f& center )
{
	m_center = center;	
}

Vector3f Camera::up() const
{
	return m_up;
}

void Camera::setUp( const Vector3f& up )
{
	m_up = up;
}

Vector3f Camera::forward() const
{
	return ( m_center - m_eye ).normalized();
}

Vector3f Camera::right() const
{
	return Vector3f::cross( forward(), m_up );
}

float Camera::zNear() const
{
	return m_zNear;
}

void Camera::setZNear( float zNear )
{
	assert( zNear > 0 );
	m_zNear = zNear;
}

float Camera::zFar() const
{
	return m_zFar;
}

void Camera::setZFar( float zFar )
{
	m_zFar = zFar;
}

Matrix4f Camera::viewMatrix() const
{
	return Matrix4f::lookAt( m_eye, m_center, m_up );
}

Matrix4f Camera::jitteredViewMatrix( float eyeX, float eyeY ) const
{
	// TODO: use Matrix4f::lookAt()

	Matrix4f view;

	// z is negative forward
	Vector3f z = -forward();
	Vector3f y = up();
	Vector3f x = right();

	// the x, y, and z vectors define the orthonormal coordinate system
	// the affine part defines the overall translation

	Vector3f jitteredEye = m_eye + eyeX * x + eyeY * y;

	view.setRow( 0, Vector4f( x, -Vector3f::dot( x, jitteredEye ) ) );
	view.setRow( 1, Vector4f( y, -Vector3f::dot( y, jitteredEye ) ) );
	view.setRow( 2, Vector4f( z, -Vector3f::dot( z, jitteredEye ) ) );
	view.setRow( 3, Vector4f( 0, 0, 0, 1 ) );

	return view;
}

Matrix4f Camera::viewProjectionMatrix() const
{
	return projectionMatrix() * viewMatrix();
}

Matrix4f Camera::inverseProjectionMatrix() const
{
	return projectionMatrix().inverse();
}

Matrix4f Camera::inverseViewMatrix() const
{
	return viewMatrix().inverse();
}

Matrix4f Camera::inverseViewProjectionMatrix() const
{
	return viewProjectionMatrix().inverse();
}

Matrix4f Camera::extrinsicsCG2CV() const
{
    return Matrix4f::ROTATE_X_180 * viewMatrix();
}

Matrix4f Camera::extrinsicsCV() const
{
    return extrinsicsCG2CV() * Matrix4f::ROTATE_X_180;
}

Matrix3f Camera::intrinsicsCV( const Vector2f& screenSize ) const
{
    float fx = screenSize.x * m_zNear / ( m_right - m_left );
    float fy = screenSize.y * m_zNear / ( m_top - m_bottom );
    float cx = 0.5f * screenSize.x + 0.5f * screenSize.x * ( m_left + m_right ) / ( m_right - m_left );
    float cy = 0.5f * screenSize.y + 0.5f * screenSize.y * ( m_bottom + m_top ) / ( m_top - m_bottom );

    return Matrix3f
    (
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
    );
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

	float xNumerator = xClip + depth * ( m_right + m_left ) / ( m_right - m_left );
	float yNumerator = yClip + depth * ( m_top + m_bottom ) / ( m_top - m_bottom );

	float xDenominator = ( 2 * m_zNear ) / ( m_right - m_left );
	float yDenominator = ( 2 * m_zNear ) / ( m_top - m_bottom );

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
	// convert from screen coordinates to NDC
	float ndcX = 2 * ( xy.x - viewport.origin().x ) / viewport.width() - 1;
	float ndcY = 2 * ( xy.y - viewport.origin().y ) / viewport.height() - 1;

	Vector4f clip( ndcX, ndcY, 0, 1 );
	Vector4f eye = inverseProjectionMatrix() * clip;
	Vector4f world = inverseViewMatrix() * eye;

	Vector3f pointOnNearPlane = world.homogenized().xyz;

	// TODO: can use pixelToWorld on z = zNear(), but pixelToWorld needs a viewport version

	return ( pointOnNearPlane - m_eye ).normalized();
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
void Camera::copyLookAt( const Camera& from, Camera& to )
{
    to.m_eye = from.m_eye;
    to.m_center = from.m_center;
    to.m_up = from.m_up;
}
