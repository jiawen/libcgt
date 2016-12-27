#pragma once

#include <string>

#include "libcgt/core/cameras/Camera.h"
#include "libcgt/core/vecmath/Matrix4f.h"
#include "libcgt/core/vecmath/Vector4f.h"

class OrthographicCamera : public Camera
{
public:

    using GLFrustum = libcgt::core::cameras::GLFrustum;

    OrthographicCamera( const Vector3f& eye = Vector3f( 0, 0, 5 ),
        const Vector3f& center = Vector3f( 0, 0, 0 ),
        const Vector3f& up = Vector3f( 0, 1, 0 ),
        const GLFrustum& frustum =
            GLFrustum{ -5.0f, 5.0f, -5.0f, 5.0f, -1.0f, 1.0f } );

    virtual Matrix4f projectionMatrix() const override;

    virtual std::string toString() const override;

private:

};

// TODO: implement lerp(). Implement a generic cubic interpolate based
//   on lerp().
