#include "cameras/OrthographicCamera.h"

#include <sstream>

#include <math/MathUtils.h>
#include <vecmath/Quat4f.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

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
    return Matrix4f::orthographicProjection
    (
        m_frustum.left, m_frustum.right,
        m_frustum.bottom, m_frustum.top,
        m_frustum.zNear, m_frustum.zFar,
        m_directX
    );
}

// virtual
std::string OrthographicCamera::toString() const
{
    std::stringstream sstream;

    sstream << "eye " << m_eye.toString() << "\n";
    sstream << "center " << m_center.toString() << "\n";
    sstream << "up " << m_up.toString() << "\n";
    sstream << "left " << m_frustum.left << "\n";
    sstream << "right " << m_frustum.right << "\n";
    sstream << "bottom " << m_frustum.bottom << "\n";
    sstream << "top " << m_frustum.top << "\n";
    sstream << "zNear " << m_frustum.zNear << "\n";
    sstream << "zFar " << m_frustum.zFar << "\n";

    return sstream.str();
}