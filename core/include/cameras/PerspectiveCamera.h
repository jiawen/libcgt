#ifndef PERSPECTIVE_CAMERA_H
#define PERSPECTIVE_CAMERA_H

#include "Camera.h"

#include <QString>

class PerspectiveCamera : public Camera
{
public:

	// fovY: field of view angle in the y direction, in degrees
	// aspect: aspect ratio in width over height (i.e. x over y)
	PerspectiveCamera( const Vector3f& eye = Vector3f( 0, 0, 5 ),
		const Vector3f& center = Vector3f( 0, 0, 0 ),
		const Vector3f& up = Vector3f( 0, 1, 0 ),
		float fovY = 50.0f, float aspect = 1.0f,
		float zNear = 1.0f, float zFar = 100.0f,
		bool zFarIsInfinite = false,
		bool isDirectX = true );

	// gets the parameters used to set this perspective camera
	// note that these are simply the cached values
	// the state can become *inconsistent* if GLCamera::setFrustum()
	// calls are made
	void getPerspective( float* pfFovY, float* pfAspect,
		float* pfZNear, float* pfZFar,
		bool* pbZFarIsInfinite = nullptr );

	void setPerspective( float fovY = 50.0f, float aspect = 1.0f,
		float zNear = 1.0f, float zFar = 100.0f,
		bool zFarIsInfinite = false );

	float aspect() const;
	void setAspect( float aspect );
	void setAspect( int width, int height );

	// TODO: switch to storing radians internally
	//float fovYRadians() const;
	//void setFovYRadians( float fovY );

	float fovYDegrees() const;
	void setFovYDegrees( float fovY );

	Matrix4f projectionMatrix() const;

	bool saveTXT( QString filename );
	static bool loadTXT( QString filename, PerspectiveCamera& camera );

	static PerspectiveCamera lerp( const PerspectiveCamera& c0, const PerspectiveCamera& c1, float t );
	static PerspectiveCamera cubicInterpolate( const PerspectiveCamera& c0, const PerspectiveCamera& c1, const PerspectiveCamera& c2, const PerspectiveCamera& c3, float t );	

private:

	float m_fovY;
	float m_aspect;

};

#endif // PERSPECTIVE_CAMERA_H
