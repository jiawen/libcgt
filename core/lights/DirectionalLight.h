#pragma once

#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Matrix3f.h"
#include "libcgt/core/vecmath/Matrix4f.h"

#include "libcgt/core/cameras/Camera.h"

class Box3f;

// A directional light (at infinity)
// The light direction is conventinally defined
// *from* the light, *towards* the scene
class DirectionalLight
{
public:

    Vector3f m_direction{ 0, 0, 1 };

    // Returns the basis matrix for this light
    // such that each *row* is a direction
    // rows 0 and 1 are normal to the light direction and each other
    // row 2 is the light direction
    Matrix3f lightBasis() const;

    // Returns the world -> light matrix
    // encompassing both the camera and the scene
    Matrix4f lightMatrix( const Camera& camera, const Box3f& sceneBoundingBox );
};
