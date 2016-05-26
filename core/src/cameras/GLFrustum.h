#pragma once

#include <vecmath/Matrix4f.h>

// An OpenGL-style frustum, comprising:
// near and far planes:
//   zNear and zFar are *distances*, not coordinates, in front of the center of
//     projection.
//   zNear > 0, zFar > zNear.
//   zFar can be infinity (std::numeric_limits< float >::infinity())
// left, right, bottom, top:
//   *coordinates*, not distances, specifying the field of view and aspect
//     ratio *at the near plane*.
//   Typically, left and bottom < 0, right and top > 0.

class GLFrustum
{
public:

    // Make a symmetric frustum from field of view (in radians) and screen
    // aspect ratio. This is a replacement for gluPerspective().
    static GLFrustum symmetric( float fovYRadians, float aspectRatio,
        float zNear, float zFar );

    float left;
    float right;
    float bottom;
    float top;

    float zNear;
    float zFar; // May be std::numeric_limits<float>::infinity().

    // Linear interpolation between two GLFrustum instances.
    // If either input has zFar = infinity, the result will have zFar = infinity.
    static GLFrustum lerp( const GLFrustum& f0, const GLFrustum& f1, float t );

    // Cubic interpolation between four GLFrustum instances.
    // If any input zFar = infinity, the result will have zFar = infinity.
    static GLFrustum cubicInterpolate(
        const GLFrustum& f0, const GLFrustum& f1, const GLFrustum& f2, const GLFrustum& f3,
        float t );
};

