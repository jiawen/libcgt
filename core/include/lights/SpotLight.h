#ifndef SPOT_LIGHT_H
#define SPOT_LIGHT_H

#include <QString>
#include <qvector.h>

#include <math/MathUtils.h>

#include <vecmath/Vector3f.h>
#include <vecmath/Matrix3f.h>
#include <vecmath/Matrix4f.h>

class SpotLight
{
public:

	// TODO: variable aspect ratio
	// TODO: infinite frustum
	// TODO: cache matrices

	SpotLight( const Vector3f& position = Vector3f( 0, 5, 0 ),
		const Vector3f& center = Vector3f( 0, 0, 0 ),
		const Vector3f& up = Vector3f( 0, 0, 1 ),
		float fovYRadians = MathUtils::degreesToRadians( 50.f ),
		float zNear = 1.0f, float fZFar = 100.0f );

	void setLookAt( const Vector3f& position,
		const Vector3f& center,
		const Vector3f& up );

	const Vector3f& position() const;
	void setPosition( const Vector3f& position );

	const Vector3f& center() const;
	void setCenter( const Vector3f& center );

	const Vector3f& up() const;
	void setUp( const Vector3f& up );

	float fovYRadians() const;
	void setFovYRadians( float fov );

	float aspectRatio() const;
	void setAspectRatio( float a );

	float zNear() const;
	void setZNear( float zNear );

	float zFar() const;
	void setZFar( float zFar );

	Vector3f lightDirection() const;

	Vector3f right() const;

	// world --> clip
	Matrix4f lightProjectionMatrix() const;

	// clip --> world
	Matrix4f inverseLightProjectionMatrix() const;

	// world --> light
	Matrix4f lightMatrix() const;

	// light --> world
	Matrix4f inverseLightMatrix() const;

	// light --> clip
	Matrix4f projectionMatrix() const;

	// gives the vertices of a rectangle z units in front of the light position
	void rectangleAlignedAt( float z,
		Vector3f* bottomLeft, Vector3f* bottomRight,
		Vector3f* topRight, Vector3f* topLeft ) const;

	QVector< Vector3f > getFrustumCorners() const;

	bool saveTXT( QString filename );

private:

	Vector3f m_position;
	Vector3f m_center;
	Vector3f m_up;

	float m_fovYRadians;
	float m_zNear;
	float m_zFar;

	float m_aspect;
};

#endif // SPOT_LIGHT_H
