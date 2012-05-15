#pragma once

#include "Camera.h"

#include <QString>

class PerspectiveCamera : public Camera
{
public:
	
	// fovY: field of view angle in the y direction, in degrees
	// aspect: aspect ratio in width over height
	PerspectiveCamera( const Vector3f& eye = Vector3f( 0, 0, 5 ),
		const Vector3f& center = Vector3f( 0, 0, 0 ),
		const Vector3f& up = Vector3f( 0, 1, 0 ),
		float fovYDegrees = 50.0f, float aspect = 1.0f,
		float zNear = 1.0f, float zFar = 100.0f,
		bool zFarIsInfinite = false,
		bool isDirectX = true );

	// gets the parameters used to set this perspective camera
	// note that these are simply the cached values
	// the state can become *inconsistent* if GLCamera::setFrustum()
	// calls are made
	//
	// TODO: maybe a PerspectiveCamera should disallow setting of top, right, etc
	// a skewed perspective camera for depth of field should return a general camera...
	void getPerspective( float* pfFovYDegrees, float* pfAspect,
		float* pfZNear, float* pfZFar,
		bool* pbZFarIsInfinite = nullptr );

	void setPerspective( float fovYDegrees = 50.0f, float aspect = 1.0f,
		float zNear = 1.0f, float zFar = 100.0f,
		bool zFarIsInfinite = false );

	// get/set the aspect ratio of the screen,
	// defined as width / height
	float aspect() const;
	void setAspect( float aspect );
	void setAspect( int width, int height );

	// get/set the field of view, in radians
	float fovXRadians() const;
	float fovYRadians() const;
	void setFovYRadians( float fovY );

	// returns half the field of view, in radians
	// very useful in projection math
	float halfFovXRadians() const;
	float halfFovYRadians() const;

	// returns tangent of half the field of view
	float tanHalfFovX() const;
	float tanHalfFovY() const;

	// get/set the field of view, in degree
	float fovYDegrees() const;
	void setFovYDegrees( float fovY );

	Matrix4f projectionMatrix() const;

	// TODO: harmonize this with math, maybe just make these static functions
	// for a perspective camera, you only need the field of view and viewport aspect ratio
	// for a skewed camera, you need left/right/bottom/top, along with zNear
	// (or any focal plane in general that's aligned with the view direction,
	// but the l/r/b/t need to be defined on that plane)
	// 
	// TODO: harmonize the aspect ratio: the viewport does not have to be the same aspect ratio as that of the camera

	// given a pixel (x,y) in screen space (in [0,screenSize.x), [0,screenSize.y))
	// and an actual depth value (\in [0, +inf)), not distance along ray,
	// returns its eye space coordinates (right handed, output z will be negative), w = 1
	virtual Vector4f pixelToEye( const Vector2f& xy, float depth, const Vector2i& screenSize );

	bool saveTXT( QString filename );
	
	// TODO: careful!  when we serialize in degrees
	static bool loadTXT( QString filename, PerspectiveCamera& camera );

	static PerspectiveCamera lerp( const PerspectiveCamera& c0, const PerspectiveCamera& c1, float t );
	static PerspectiveCamera cubicInterpolate( const PerspectiveCamera& c0, const PerspectiveCamera& c1, const PerspectiveCamera& c2, const PerspectiveCamera& c3, float t );	

private:

	void updateFrustum();

	float m_fovYRadians;
	float m_aspect;

};
