#include "cameras/Camera.h"

#include <cmath>
#include <cstdio>

#include <math/MathUtils.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Quat4f.h>

#include "geometry/BoundingBox3f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Camera::Camera( const Vector3f& eye, const Vector3f& center, const Vector3f& up,
	float left, float right,
	float bottom, float top,
	float zNear, float zFar, bool zFarIsInfinite,
	bool isDirectX )
{
	setLookAt( eye, center, up );
	setFrustum
	(
		left, right,
		bottom, top,
		zNear, zFar, zFarIsInfinite
	);
	setDirectX( isDirectX );
}

void Camera::setDirectX( bool directX )
{
	m_directX = directX;
}

void Camera::getFrustum( float* pfLeft, float* pfRight,
	float* pfBottom, float* pfTop,
	float* pfZNear, float* pfZFar,
	bool* pbZFarIsInfinite ) const
{
	*pfLeft = m_left;
	*pfRight = m_right;

	*pfBottom = m_bottom;
	*pfTop = m_top;

	*pfZNear = m_zNear;
	*pfZFar = m_zFar;

	if( pbZFarIsInfinite != NULL )
	{
		*pbZFarIsInfinite = m_zFarIsInfinite;
	}
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

		corners[ i ] = ( invViewProj * cubePoint ).homogenized().xyz();
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

bool Camera::isZFarInfinite()
{
	return m_zFarIsInfinite;
}

void Camera::setFrustum( float left, float right,
	float bottom, float top,
	float zNear, float zFar,
	bool zFarIsInfinite )
{
	m_left = left;
	m_right = right;

	m_bottom = bottom;
	m_top = top;

	m_zNear = zNear;
	m_zFar = zFar;

	m_zFarIsInfinite = zFarIsInfinite;
}

void Camera::getLookAt( Vector3f* pEye,
	Vector3f* pCenter,
	Vector3f* pUp ) const
{
	*pEye = m_eye;
	*pCenter = m_center;
	*pUp = m_up;
}

void Camera::setLookAt( const Vector3f& eye,
	const Vector3f& center,
	const Vector3f& up )
{
	m_eye = eye;
	m_center = center;
	m_up = up;

#if 0
	m_eye = eye;
	m_center = center;
	m_up = up.normalized();
	
	// recompute up to ensure an orthonormal basis
	m_up = Vector3f::cross( -forward(), right() );
#endif
}

void Camera::setEye( const Vector3f& eye )
{
	m_eye = eye;
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

Matrix4f Camera::jitteredProjectionMatrix( float eyeX, float eyeY, float focusZ ) const
{
	float dx = -eyeX * m_zNear / focusZ;
	float dy = -eyeY * m_zNear / focusZ;

	if( m_zFarIsInfinite )
	{
		return Matrix4f::infinitePerspectiveProjection( m_left + dx, m_right + dx,
			m_bottom + dy, m_top + dy,
			m_zNear, m_directX );
	}
	else
	{
		return Matrix4f::perspectiveProjection( m_left + dx, m_right + dx,
			m_bottom + dy, m_top + dy,
			m_zNear, m_zFar, m_directX );
	}
}

Matrix4f Camera::viewMatrix() const
{
	return Matrix4f::lookAt( m_eye, m_center, m_up );
}

Matrix4f Camera::jitteredViewMatrix( float eyeX, float eyeY ) const
{
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

Matrix4f Camera::jitteredViewProjectionMatrix( float eyeX, float eyeY, float focusZ ) const
{
	return
	(
		jitteredProjectionMatrix( eyeX, eyeY, focusZ ) *
		jitteredViewMatrix( eyeX, eyeY )
	);
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

Vector3f Camera::pixelToDirection( const Vector2f& xy, const Vector2i& screenSize )
{
	return pixelToDirection( xy, Rect2f( screenSize.x, screenSize.y ) );
}

Vector3f Camera::pixelToDirection( const Vector2f& xy, const Rect2f& viewport )
{
	// convert from screen coordinates to NDC
	float ndcX = 2 * ( xy.x - viewport.origin().x ) / viewport.width() - 1;
	float ndcY = 2 * ( xy.y - viewport.origin().y ) / viewport.height() - 1;

	Vector4f clip( ndcX, ndcY, 0, 1 );
	Vector4f eye = inverseProjectionMatrix() * clip;
	Vector4f world = inverseViewMatrix() * eye;

	Vector3f pointOnNearPlane = world.homogenized().xyz();

	return ( pointOnNearPlane - m_eye ).normalized();
}

Vector4f Camera::projectToScreen( const Vector4f& world, const Vector2i& screenSize )
{
	Vector4f clip = viewProjectionMatrix() * world;
	Vector4f ndc = clip.homogenized();

	float sx = screenSize.x * 0.5f * ( ndc.x + 1.0f );
	float sy = screenSize.y * 0.5f * ( ndc.y + 1.0f );
	float sz;

	if( m_directX )
	{
		sz = ndc.z;
	}
	else
	{
		sz = 0.5f * ( ndc.z + 1.0f );
	}
	
	float sw = clip.w;

	return Vector4f( sx, sy, sz, sw );
}

Vector2f Camera::pixelToNDC( const Vector2f& xy, const Vector2i& screenSize )
{
	// convert from screen coordinates to NDC
	float ndcX = 2 * xy.x / screenSize.x - 1;
	float ndcY = 2 * xy.y / screenSize.y - 1;

	return Vector2f( ndcX, ndcY );
}

Vector4f Camera::pixelToEye( const Vector2f& xy, float depth, const Vector2i& screenSize )
{
	Vector2f ndcXY = pixelToNDC( xy, screenSize );

	// depth = -zEye
	// xClip = xEye * ( 2 * zNear ) / ( right - left ) + zEye * ( right + left ) / ( right - left )
	// wClip = -zEye
	// xNDC = xClip = wClip = xClip / -zEye = xClip / depth
	//	
	// -->
	// xNDC = xClip / depth
	// xEye = ( xClip + depth * ( right + left ) / ( right - left ) ) * ( right - left ) / ( 2 * zNear )

	float xClip = ndcXY.x * depth;
	float yClip = ndcXY.y * depth;

	float xEye = ( xClip + depth * ( m_right + m_left ) / ( m_right - m_left ) ) * ( m_right - m_left ) / ( 2 * m_zNear );
	float yEye = ( yClip + depth * ( m_top + m_bottom ) / ( m_top - m_bottom ) ) * ( m_top - m_bottom ) / ( 2 * m_zNear );	
	float zEye = -depth;

	return Vector4f( xEye, yEye, zEye, 1 );
}

Vector4f Camera::pixelToWorld( const Vector2f& xy, float depth, const Vector2i& screenSize )
{
	Vector4f eye = pixelToEye( xy, depth, screenSize );
	return inverseViewMatrix() * eye;
}
