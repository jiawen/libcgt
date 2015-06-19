#pragma once

#include "Camera.h"
#include "vecmath/Matrix4f.h"
#include "vecmath/Vector4f.h"

class QString;

class OrthographicCamera : public Camera
{
public:

	OrthographicCamera( const Vector3f& eye = Vector3f( 0, 0, 5 ),
		const Vector3f& center = Vector3f( 0, 0, 0 ),
		const Vector3f& up = Vector3f( 0, 1, 0 ),
        const GLFrustum& frustum = { -5.0f, 5.0f, -5.0f, 5.0f, -1.0f, 1.0f } );

	virtual Matrix4f projectionMatrix() const;

	bool saveTXT( QString filename );

	//static PerspectiveCamera cubicInterpolate( const PerspectiveCamera& c0, const PerspectiveCamera& c1, const PerspectiveCamera& c2, const PerspectiveCamera& c3, float t );	

private:

};
