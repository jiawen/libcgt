#pragma once

#include <vecmath/Vector3f.h>
#include <vecmath/Matrix3f.h>
#include <vecmath/Matrix4f.h>

#include "cameras/Camera.h"

class BoundingBox3f;

// A directional light (at infinity)
// The light direction is conventinally defined
// *from* the light, *towards* the scene
class DirectionalLight
{
public:

	// The default directional light,
	// with lightDirection (0,0,1)
	DirectionalLight();

	DirectionalLight( const Vector3f& direction );

	Vector3f direction() const;
	void setDirection( const Vector3f& direction );

	// Returns the basis matrix for this light
	// such that each *row* is a direction
	// rows 0 and 1 are normal to the light direction and each other
	// row 2 is the light direction	
	Matrix3f lightBasis() const;

	// Returns the world -> light matrix
	// encompassing both the camera and the scene	
	Matrix4f lightMatrix( const Camera& camera, const BoundingBox3f& sceneBoundingBox );

private:

	Vector3f m_direction;

};
