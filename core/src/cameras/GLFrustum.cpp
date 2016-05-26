#include "cameras/GLFrustum.h"

#include <math/MathUtils.h>

using libcgt::core::math::cubicInterpolate;
using libcgt::core::math::lerp;

// static
GLFrustum GLFrustum::symmetric( float fovYRadians, float aspectRatio,
    float zNear, float zFar )
{
    GLFrustum frustum;

    float tanHalfFovY = tan( 0.5f * fovYRadians );

    // tan( theta / 2 ) = up / zNear
    frustum.top = zNear * tanHalfFovY;
    frustum.bottom = -frustum.top;

    // aspect = width / height = ( right - left ) / ( top - bottom )
    frustum.right = aspectRatio * frustum.top;
    frustum.left = -frustum.right;

    frustum.zNear = zNear;
    frustum.zFar = zFar;

    return frustum;
}

// static
GLFrustum GLFrustum::lerp( const GLFrustum& f0, const GLFrustum& f1, float t )
{
    GLFrustum frustum;

    frustum.left = ::lerp( f0.left, f1.left, t );
    frustum.right = ::lerp( f0.right, f1.right, t );
    frustum.bottom = ::lerp( f0.bottom, f1.bottom, t );
    frustum.top = ::lerp( f0.top, f1.top, t );

    frustum.zNear = ::lerp( f0.zNear, f1.zNear, t );

    if( isinf( f0.zFar ) || isinf( f1.zFar ) )
    {
        frustum.zFar = std::numeric_limits< float >::infinity();
    }
    else
    {
        frustum.zFar = ::lerp( f0.zFar, f1.zFar, t );
    }

    return frustum;
}

// static
GLFrustum GLFrustum::cubicInterpolate(
    const GLFrustum& f0, const GLFrustum& f1, const GLFrustum& f2, const GLFrustum& f3,
    float t )
{
    GLFrustum frustum;

    frustum.left = ::cubicInterpolate( f0.left, f1.left, f2.left, f3.left, t );
    frustum.right = ::cubicInterpolate( f0.right, f1.right, f2.right, f3.right, t );
    frustum.bottom = ::cubicInterpolate( f0.bottom, f1.bottom, f2.bottom, f3.bottom, t);
    frustum.top = ::cubicInterpolate( f0.top, f1.top, f2.top, f3.top, t );

    frustum.zNear = ::cubicInterpolate( f0.zNear, f1.zNear, f2.zNear, f3.zNear, t );

    if( isinf( f0.zFar ) || isinf( f1.zFar ) || isinf( f2.zFar ) || isinf( f3.zFar ) )
    {
        frustum.zFar = std::numeric_limits< float >::infinity();
    }
    else
    {
        frustum.zFar = ::cubicInterpolate( f0.zFar, f1.zFar, f2.zFar, f3.zFar, t );
    }

    return frustum;
}
