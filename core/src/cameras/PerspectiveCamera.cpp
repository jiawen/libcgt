#include "cameras/PerspectiveCamera.h"

#include <QFile>
#include <QTextStream>

#include <math/Arithmetic.h>
#include <math/MathUtils.h>
#include <vecmath/Quat4f.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

PerspectiveCamera::PerspectiveCamera( const Vector3f& eye, const Vector3f& center, const Vector3f& up,
	float fovY, float aspect,
	float zNear, float zFar,
	bool zFarIsInfinite,
	bool isDirectX )
{
	setPerspective( fovY, aspect, zNear, zFar, zFarIsInfinite );
	setLookAt( eye, center, up );
	setDirectX( isDirectX );
}

void PerspectiveCamera::getPerspective( float* pfFovY, float* pfAspect,
	float* pfZNear, float* pfZFar, bool* pbZFarIsInfinite )
{
	*pfFovY = m_fovY;
	*pfAspect = m_aspect;

	*pfZNear = m_zNear;
	*pfZFar = m_zFar;

	if( pbZFarIsInfinite != nullptr )
	{
		*pbZFarIsInfinite = m_zFarIsInfinite;
	}
}

void PerspectiveCamera::setPerspective( float fovY, float aspect,
	float zNear, float zFar, bool zFarIsInfinite )
{
	// store fov and aspect ratio parameters
	m_fovY = fovY;
	m_aspect = aspect;

	m_zNear = zNear;
	m_zFar = zFar;
	m_zFarIsInfinite = zFarIsInfinite;

	updateFrustum();	
}

float PerspectiveCamera::aspect() const
{
	return m_aspect;
}

void PerspectiveCamera::setAspect( float aspect )
{
	m_aspect = aspect;
	updateFrustum();
}

void PerspectiveCamera::setAspect( int width, int height )
{
	setAspect( Arithmetic::divideIntsToFloat( width, height ) );
}

float PerspectiveCamera::fovYRadians() const
{
	return MathUtils::degreesToRadians( m_fovY );
}

void PerspectiveCamera::setFovYRadians( float fovY )
{
	m_fovY = MathUtils::radiansToDegrees( fovY );
	updateFrustum();
}

float PerspectiveCamera::halfFovYRadians() const
{
	return 0.5f * fovYRadians();
}

float PerspectiveCamera::tanHalfFovY() const
{
	return tan( halfFovYRadians() );
}

float PerspectiveCamera::fovYDegrees() const
{
	return m_fovY;
}

void PerspectiveCamera::setFovYDegrees( float fovY )
{
	m_fovY = fovY;

	updateFrustum();
}

// virtual
Matrix4f PerspectiveCamera::projectionMatrix() const
{
	if( m_zFarIsInfinite )
	{
		return Matrix4f::infinitePerspectiveProjection( m_left, m_right,
			m_bottom, m_top,
			m_zNear, m_directX );
	}
	else
	{
		return Matrix4f::perspectiveProjection( m_left, m_right,
			m_bottom, m_top,
			m_zNear, m_zFar, m_zFarIsInfinite );
	}
}

Vector4f PerspectiveCamera::pixelToEye( const Vector2f& xy, float depth, const Vector2i& screenSize )
{
	Vector2f ndcXY = pixelToNDC( xy, screenSize );
	float t = tanHalfFovY();
	// x_ndc = x_eye / tan( theta/2 ) / depth / aspect
	// y_ndc = y_eye / tan( theta/2 ) / depth
	
	float xEye = ndcXY.x * t * aspect() * depth;
	float yEye = ndcXY.y * t * depth;
	float zEye = -depth; // right handed, z points toward viewer

	return Vector4f( xEye, yEye, zEye, 1 );
}

// static
bool PerspectiveCamera::loadTXT( QString filename, PerspectiveCamera& camera )
{
	QFile inputFile( filename );

	// try to open the file in write only mode
	if( !( inputFile.open( QIODevice::ReadOnly ) ) )
	{
		return false;
	}

	QTextStream inputTextStream( &inputFile );
	inputTextStream.setCodec( "UTF-8" );

	QString str;
	int i;

	Vector3f eye;
	Vector3f center;
	Vector3f up;
	float zNear;
	float zFar;
	float fovY;
	float aspect;

	bool isInfinite;
	bool isDirectX;

	inputTextStream >> str >> eye[ 0 ] >> eye[ 1 ] >> eye[ 2 ];
	inputTextStream >> str >> center[ 0 ] >> center[ 1 ] >> center[ 2 ];
	inputTextStream >> str >> up[ 0 ] >> up[ 1 ] >> up[ 2 ];
	inputTextStream >> str >> zNear;
	inputTextStream >> str >> zFar;
	inputTextStream >> str >> i;
	isInfinite = ( i != 0 );
	inputTextStream >> str >> fovY;
	inputTextStream >> str >> aspect;
	inputTextStream >> str >> i;
	isDirectX = ( i != 0 );

	inputFile.close();

	camera.setLookAt( eye, center, up );
	camera.setPerspective( fovY, aspect, zNear, zFar, isInfinite );
	camera.setDirectX( isDirectX );

	return true;
}

bool PerspectiveCamera::saveTXT( QString filename )
{
	QFile outputFile( filename );

	// try to open the file in write only mode
	if( !( outputFile.open( QIODevice::WriteOnly ) ) )
	{
		return false;
	}

	QTextStream outputTextStream( &outputFile );
	outputTextStream.setCodec( "UTF-8" );

	outputTextStream << "eye " << m_eye[ 0 ] << " " << m_eye[ 1 ] << " " << m_eye[ 2 ] << "\n";
	outputTextStream << "center " << m_center[ 0 ] << " " << m_center[ 1 ] << " " << m_center[ 2 ] << "\n";
	outputTextStream << "up " << m_up[ 0 ] << " " << m_up[ 1 ] << " " << m_up[ 2 ] << "\n";
	outputTextStream << "zNear " << m_zNear << "\n";
	outputTextStream << "zFar " << m_zFar << "\n";
	outputTextStream << "zFarInfinite " << static_cast< int >( m_zFarIsInfinite ) << "\n";
	outputTextStream << "fovY " << m_fovY << "\n";
	outputTextStream << "aspect " << m_aspect << "\n";
	outputTextStream << "isDirectX " << static_cast< int >( m_directX ) << "\n";

	outputFile.close();
	return true;
}

// static
PerspectiveCamera PerspectiveCamera::lerp( const PerspectiveCamera& c0, const PerspectiveCamera& c1, float t )
{
	float fov = MathUtils::lerp( c0.m_fovY, c1.m_fovY, t );
	float aspect = MathUtils::lerp( c0.m_aspect, c1.m_aspect, t );

	float zNear = MathUtils::lerp( c0.m_zNear, c1.m_zNear, t );
	float zFar = MathUtils::lerp( c0.m_zFar, c1.m_zFar, t );

	bool farIsInfinite = c0.m_zFarIsInfinite;
	bool isDirectX = c0.m_directX;

	Vector3f position = Vector3f::lerp( c0.m_eye, c1.m_eye, t );

	Quat4f q0 = Quat4f::fromRotatedBasis( c0.right(), c0.up(), -( c0.forward() ) );	
	Quat4f q1 = Quat4f::fromRotatedBasis( c1.right(), c1.up(), -( c1.forward() ) );	
	Quat4f q = Quat4f::slerp( q0, q1, t );

	Vector3f x = q.rotateVector( Vector3f::RIGHT );
	Vector3f y = q.rotateVector( Vector3f::UP );
	Vector3f z = q.rotateVector( -Vector3f::FORWARD );

	Vector3f center = position - z;

	PerspectiveCamera camera
	(
		position, center, y,
		fov, aspect,
		zNear, zFar, farIsInfinite
	);
	camera.m_directX = isDirectX;

	return camera;
}


// static
PerspectiveCamera PerspectiveCamera::cubicInterpolate( const PerspectiveCamera& c0, const PerspectiveCamera& c1, const PerspectiveCamera& c2, const PerspectiveCamera& c3, float t )
{
	float fov = MathUtils::cubicInterpolate( c0.m_fovY, c1.m_fovY, c2.m_fovY, c3.m_fovY, t );
	float aspect = MathUtils::cubicInterpolate( c0.m_aspect, c1.m_aspect, c2.m_aspect, c3.m_aspect, t );

	float zNear = MathUtils::cubicInterpolate( c0.m_zNear, c1.m_zNear, c2.m_zNear, c3.m_zNear, t );
	float zFar = MathUtils::cubicInterpolate( c0.m_zFar, c1.m_zFar, c2.m_zFar, c3.m_zFar, t );

	bool farIsInfinite = c0.m_zFarIsInfinite;
	bool isDirectX = c0.m_directX;

	Vector3f position = Vector3f::cubicInterpolate( c0.m_eye, c1.m_eye, c2.m_eye, c3.m_eye, t );

	Quat4f q0 = Quat4f::fromRotatedBasis( c0.right(), c0.up(), -( c0.forward() ) );	
	Quat4f q1 = Quat4f::fromRotatedBasis( c1.right(), c1.up(), -( c1.forward() ) );	
	Quat4f q2 = Quat4f::fromRotatedBasis( c2.right(), c2.up(), -( c2.forward() ) );	
	Quat4f q3 = Quat4f::fromRotatedBasis( c3.right(), c3.up(), -( c3.forward() ) );	

	Quat4f q = Quat4f::cubicInterpolate( q0, q1, q2, q3, t );

	Vector3f x = q.rotateVector( Vector3f::RIGHT );
	Vector3f y = q.rotateVector( Vector3f::UP );
	Vector3f z = q.rotateVector( -Vector3f::FORWARD );

	Vector3f center = position - z;

	PerspectiveCamera camera
	(
		position, center, y,
		fov, aspect,
		zNear, zFar, farIsInfinite
	);
	camera.m_directX = isDirectX;

	return camera;
}

void PerspectiveCamera::updateFrustum()
{
	// tan( theta / 2 ) = up / zNear
	float top = m_zNear * tanHalfFovY();
	float bottom = -top;

	// aspect = width / height = ( right - left ) / ( top - bottom )
	float right = m_aspect * top;
	float left = -right;

	setFrustum( left, right, bottom, top, m_zNear, m_zFar, m_zFarIsInfinite );
}
