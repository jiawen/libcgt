#include "libcgt/core/cameras/OrthographicCamera.h"

#include <sstream>

#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Quat4f.h"

using libcgt::core::cameras::GLFrustum;

OrthographicCamera::OrthographicCamera( const Vector3f& eye,
    const Vector3f& center,
    const Vector3f& up,
    const GLFrustum& frustum )
{
    setFrustum( frustum );
    setLookAt( eye, center, up );
}

// virtual
Matrix4f OrthographicCamera::projectionMatrix() const
{
    GLFrustum f = frustum();

    return Matrix4f::orthographicProjection
    (
        f.left, f.right,
        f.bottom, f.top,
        f.zNear, f.zFar,
        isDirectX()
    );
}

// virtual
std::string OrthographicCamera::toString() const
{
    GLFrustum f = frustum();

    std::ostringstream sstream;

    sstream << "orthographic_camera" << "\n";
    sstream << "eye " << eye().toString() << "\n";
    sstream << "right " << right().toString() << "\n";
    sstream << "up " << up().toString() << "\n";
    sstream << "back " << back().toString() << "\n";
    sstream << "frustum_left " << f.left << "\n";
    sstream << "frustum_right " << f.right << "\n";
    sstream << "frustum_bottom " << f.bottom << "\n";
    sstream << "frustum_top " << f.top << "\n";
    sstream << "frustum_zNear " << f.zNear << "\n";
    sstream << "frustum_zFar " << f.zFar << "\n";

    return sstream.str();
}
