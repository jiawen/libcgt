#pragma once

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

struct GLFrustum
{
    float left;
    float right;
    float bottom;
    float top;

    float zNear;
    float zFar;
};
