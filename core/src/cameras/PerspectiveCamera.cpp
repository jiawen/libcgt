#include "cameras/PerspectiveCamera.h"

#include <cmath>
#include <limits>

// TODO: remove Qt
#include <QFile>
#include <QString>
#include <QTextStream>

#include "cameras/CameraUtils.h"
#include "math/Arithmetic.h"
#include "math/MathUtils.h"
#include "vecmath/Quat4f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
const PerspectiveCamera PerspectiveCamera::CANONICAL(
	Vector3f( 0, 0, 0 ), Vector3f( 0, 0, -1 ), Vector3f( 0, 1, 0 ),
	MathUtils::HALF_PI, 1.0f, 1.0f, 100.0f );

// static
const PerspectiveCamera PerspectiveCamera::FRONT(
	Vector3f( 0, 0, 5 ), Vector3f( 0, 0, 0 ), Vector3f( 0, 1, 0 ),
	MathUtils::degreesToRadians( 50.0f ), 1.0f, 1.0f, 100.0f );

// static
const PerspectiveCamera PerspectiveCamera::RIGHT(
    Vector3f( 5, 0, 0 ), Vector3f( 0, 0, 0 ), Vector3f( 0, 1, 0 ),
    MathUtils::degreesToRadians( 50.0f ), 1.0f, 1.0f, 100.0f );

// static
const PerspectiveCamera PerspectiveCamera::TOP(
    Vector3f( 0, 5, 0 ), Vector3f( 0, 0, 0 ), Vector3f( 0, 0, -1 ),
    MathUtils::degreesToRadians( 50.0f ), 1.0f, 1.0f, 100.0f );

PerspectiveCamera::PerspectiveCamera( const Vector3f& eye, const Vector3f& center, const Vector3f& up,
	float fovYRadians, float aspect,
	float zNear, float zFar,
	bool isDirectX )
{
	setPerspective( fovYRadians, aspect, zNear, zFar );
	setLookAt( eye, center, up );
	setDirectX( isDirectX );
}

void PerspectiveCamera::getPerspective( float& fovYRadians, float aspect,
	float& zNear, float& zFar )
{
    fovYRadians = m_fovYRadians;
    aspect = m_aspect;

    zNear = m_zNear;
    zFar = m_zFar;
}

void PerspectiveCamera::setPerspective( float fovYRadians, float aspect,
	float zNear, float zFar )
{
	// store fov and aspect ratio parameters
	m_fovYRadians = fovYRadians;
	m_aspect = aspect;

	m_zNear = zNear;
	m_zFar = zFar;

	updateFrustum();	
}

void PerspectiveCamera::setPerspectiveFromIntrinsics( const Vector2f& focalLengthPixels,
	const Vector2f& imageSize )
{
	m_fovYRadians = CameraUtils::focalLengthPixelsToFoVRadians( focalLengthPixels.y, imageSize.y );	
	m_aspect = imageSize.x / imageSize.y;

	updateFrustum();
}

void PerspectiveCamera::setPerspectiveFromIntrinsics( const Vector2f& focalLengthPixels,
	const Vector2f& imageSize,
	float zNear, float zFar )
{
	m_zNear = zNear;
	m_zFar = zFar;

	setPerspectiveFromIntrinsics( focalLengthPixels, imageSize );
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

void PerspectiveCamera::setAspect( const Vector2f& screenSize )
{
	setAspect( Arithmetic::divideIntsToFloat( screenSize.x, screenSize.y ) );
}

float PerspectiveCamera::fovXRadians() const
{
	return 2 * halfFovXRadians();	
}

float PerspectiveCamera::fovYRadians() const
{
	return m_fovYRadians;	
}

void PerspectiveCamera::setFovYRadians( float fovY )
{
	m_fovYRadians = fovY;
	updateFrustum();
}

float PerspectiveCamera::fovXDegrees() const
{
	return MathUtils::radiansToDegrees( fovXRadians() );
}

float PerspectiveCamera::fovYDegrees() const
{
	return MathUtils::radiansToDegrees( m_fovYRadians );
}

void PerspectiveCamera::setFovYDegrees( float fovY )
{
	m_fovYRadians = MathUtils::degreesToRadians( fovY );

	updateFrustum();
}

float PerspectiveCamera::halfFovXRadians() const
{
	return atan( tanHalfFovX() );
}

float PerspectiveCamera::halfFovYRadians() const
{
	return 0.5f * fovYRadians();
}

float PerspectiveCamera::tanHalfFovX() const
{
	// let phi_x = half fov x
	// let phi_y = half fov y
	// then:
	// tan( phi_x ) = right / zNear
	// tan( phi_y ) = top / zNear
	// right / top = aspect
	// so:
	// tan( phi_x ) = aspect * top / zNear
	//              = aspect * tan( phi_y )

	return tanHalfFovY() * aspect();
}

float PerspectiveCamera::tanHalfFovY() const
{
	return tan( halfFovYRadians() );
}

// virtual
void PerspectiveCamera::setZNear( float zNear )
{
    Camera::setZNear( zNear );
    updateFrustum();
}

Vector2f PerspectiveCamera::focalLengthPixels( const Vector2f& screenSize ) const
{
    float fx = CameraUtils::fovRadiansToFocalLengthPixels( fovXRadians(), screenSize.x );
    float fy = CameraUtils::fovRadiansToFocalLengthPixels( fovYRadians(), screenSize.y );
    return{ fx, fy };
}

// virtual
Matrix4f PerspectiveCamera::projectionMatrix() const
{
	if( isinf( m_zFar ) )
	{
		return Matrix4f::infinitePerspectiveProjection( m_left, m_right,
			m_bottom, m_top,
			m_zNear, m_directX );
	}
	else
	{
		return Matrix4f::perspectiveProjection( m_left, m_right,
			m_bottom, m_top,
			m_zNear, m_zFar, m_directX );
	}
}

Matrix4f PerspectiveCamera::jitteredProjectionMatrix( float eyeX, float eyeY, float focusZ ) const
{
	float dx = -eyeX * m_zNear / focusZ;
	float dy = -eyeY * m_zNear / focusZ;

	if( isinf( m_zFar ) )
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

Matrix4f PerspectiveCamera::jitteredViewProjectionMatrix( float eyeX, float eyeY, float focusZ ) const
{
	return
	(
		jitteredProjectionMatrix( eyeX, eyeY, focusZ ) *
		jitteredViewMatrix( eyeX, eyeY )
	);
}

// virtual 
Vector4f PerspectiveCamera::screenToEye( const Vector2i& xy, float depth, const Vector2f& screenSize ) const
{
    return screenToEye( Vector2f{ xy.x + 0.5f, xy.y + 0.5f }, depth, screenSize );
}

Vector4f PerspectiveCamera::screenToEye( const Vector2f& xy, float depth, const Vector2f& screenSize ) const
{
	Vector2f ndcXY = screenToNDC( xy, screenSize );
	float t = tanHalfFovY();
	// x_ndc = x_eye / tan( theta/2 ) / depth / aspect
	// y_ndc = y_eye / tan( theta/2 ) / depth
	
	float xEye = ndcXY.x * t * aspect() * depth;
	float yEye = ndcXY.y * t * depth;
	float zEye = -depth; // right handed, z points toward viewer

	return Vector4f( xEye, yEye, zEye, 1 );
}

QString PerspectiveCamera::toString() const
{
	return QString( "Perspective camera:\neye: %1\ncenter: %2\nup:%3" )
		.arg( eye().toString() )
		.arg( center().toString() )
		.arg( up().toString() );
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

bool PerspectiveCamera::loadTXT( QString filename )
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
	float fovYRadians;
	float aspect;

	bool isDirectX;

	inputTextStream >> str >> eye[ 0 ] >> eye[ 1 ] >> eye[ 2 ];
	inputTextStream >> str >> center[ 0 ] >> center[ 1 ] >> center[ 2 ];
	inputTextStream >> str >> up[ 0 ] >> up[ 1 ] >> up[ 2 ];
	inputTextStream >> str >> zNear;
	inputTextStream >> str >> zFar;
    if( zFar < 0 )
    {
        zFar = std::numeric_limits< float >::infinity();
    }
	inputTextStream >> str >> fovYRadians;
	inputTextStream >> str >> aspect;
	inputTextStream >> str >> i;
	isDirectX = ( i != 0 );

	inputFile.close();	

	setLookAt( eye, center, up );
	setPerspective( fovYRadians, aspect, zNear, zFar );
	setDirectX( isDirectX );

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
    if( isinf( m_zFar ) )
    {
        outputTextStream << "zFar " << -1 << "\n";
    }
    else
    {
        outputTextStream << "zFar " << m_zFar << "\n";
    }
	outputTextStream << "fovYRadians " << m_fovYRadians << "\n";
	outputTextStream << "aspect " << m_aspect << "\n";
	outputTextStream << "isDirectX " << static_cast< int >( m_directX ) << "\n";

	outputFile.close();
	return true;
}

// static
PerspectiveCamera PerspectiveCamera::lerp( const PerspectiveCamera& c0, const PerspectiveCamera& c1, float t )
{
	float fov = MathUtils::lerp( c0.m_fovYRadians, c1.m_fovYRadians, t );
	float aspect = MathUtils::lerp( c0.m_aspect, c1.m_aspect, t );

	float zNear = MathUtils::lerp( c0.m_zNear, c1.m_zNear, t );
	float zFar = MathUtils::lerp( c0.m_zFar, c1.m_zFar, t );

	bool isDirectX = c0.m_directX;

	Vector3f position = MathUtils::lerp( c0.m_eye, c1.m_eye, t );

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
		zNear, zFar
	);
	camera.m_directX = isDirectX;

	return camera;
}


// static
PerspectiveCamera PerspectiveCamera::cubicInterpolate( const PerspectiveCamera& c0, const PerspectiveCamera& c1, const PerspectiveCamera& c2, const PerspectiveCamera& c3, float t )
{
	float fov = MathUtils::cubicInterpolate( c0.m_fovYRadians, c1.m_fovYRadians, c2.m_fovYRadians, c3.m_fovYRadians, t );
	float aspect = MathUtils::cubicInterpolate( c0.m_aspect, c1.m_aspect, c2.m_aspect, c3.m_aspect, t );

	float zNear = MathUtils::cubicInterpolate( c0.m_zNear, c1.m_zNear, c2.m_zNear, c3.m_zNear, t );
	float zFar = MathUtils::cubicInterpolate( c0.m_zFar, c1.m_zFar, c2.m_zFar, c3.m_zFar, t );

	bool isDirectX = c0.m_directX;

	Vector3f position = MathUtils::cubicInterpolate( c0.m_eye, c1.m_eye, c2.m_eye, c3.m_eye, t );

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
		zNear, zFar
	);
	camera.m_directX = isDirectX;

	return camera;
}

void PerspectiveCamera::copyPerspective( const PerspectiveCamera& from, PerspectiveCamera& to )
{
    to.m_fovYRadians = from.m_fovYRadians;
    to.m_aspect = from.m_aspect;
    to.m_zNear = from.m_zNear;
    to.m_zFar = from.m_zFar;

    to.updateFrustum();
}

void PerspectiveCamera::updateFrustum()
{
	// tan( theta / 2 ) = up / zNear
	float top = m_zNear * tanHalfFovY();
	float bottom = -top;

	// aspect = width / height = ( right - left ) / ( top - bottom )
	float right = m_aspect * top;
	float left = -right;

	setFrustum( left, right, bottom, top, m_zNear, m_zFar );
}
