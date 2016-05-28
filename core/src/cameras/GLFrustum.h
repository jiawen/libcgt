#pragma once

#include <vecmath/Matrix4f.h>

namespace libcgt { namespace core { namespace cameras {

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

    float left;
    float right;
    float bottom;
    float top;

    float zNear;
    float zFar; // May be std::numeric_limits<float>::infinity().

    // The image aspect ratio (width divided by height).
    // Works even for an asymmetric frustum.
    float aspectRatio() const;

    // The full horizontal field of view in radians.
    // Works even for an asymmetric frustum.
    float fovXRadians() const;

    // The full vertical field of view in radians.
    // Works even for an asymmetric frustum.
    float foVYRadians() const;

    // Make an asymmetric frustum from 4 fields of view (in radians).
    // This fully determines the aspect ratio.
    //
    // To be fully consistent, leftFovRadians and bottomFoVRadians are
    // typically *negative* numbers unless the frustum is extremely skewed.
    static GLFrustum makeAsymmetricPerspective(
        float leftFovRadians, float rightFoVRadians,
        float bottomFoVRadians, float topFoVRadians,
        float zNear, float zFar );

    // Make a symmetric frustum from field of view (in radians) and image
    // aspect ratio. This is a replacement for gluPerspective().
    static GLFrustum makeSymmetricPerspective( float fovYRadians, float aspectRatio,
        float zNear, float zFar);

    // Linear interpolation between two GLFrustum instances.
    // If either input has zFar = infinity, the result will have zFar = infinity.
    static GLFrustum lerp( const GLFrustum& f0, const GLFrustum& f1, float t );

    // Cubic interpolation between four GLFrustum instances.
    // If any input zFar = infinity, the result will have zFar = infinity.
    static GLFrustum cubicInterpolate(
        const GLFrustum& f0, const GLFrustum& f1, const GLFrustum& f2, const GLFrustum& f3,
        float t );
};

} } } // cameras, core, libcgt
