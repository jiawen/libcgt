#include "libcgt/core/cameras/GLFrustum.h"

#include "libcgt/core/math/MathUtils.h"

using libcgt::core::math::cubicInterpolate;
using libcgt::core::math::lerp;

namespace libcgt { namespace core { namespace cameras {

GLFrustum::GLFrustum( float _left, float _right, float _bottom, float _top,
    float _zNear, float _zFar ) :
    left( _left ),
    right( _right ),
    bottom( _bottom ),
    top( _top ),
    zNear( _zNear ),
    zFar( _zFar )
{

}

float GLFrustum::aspectRatio() const
{
    return ( right - left ) / ( top - bottom );
}

float GLFrustum::fovXRadians() const
{
    // The left half of the fov is probably negative unless the frustum is
    // extremely skewed. The total is still right - left.
    float leftHalfFovX = atan( left / zNear );
    float rightHalfFovX = atan( right / zNear );
    return( rightHalfFovX - leftHalfFovX );
}

float GLFrustum::foVYRadians() const
{
    float bottomHalfFovX = atan( bottom / zNear );
    float topHalfFovX = atan( top / zNear );
    return( topHalfFovX - bottomHalfFovX );
}

// static
GLFrustum GLFrustum::makeAsymmetricPerspective(
    float leftFovRadians, float rightFoVRadians,
    float bottomFoVRadians, float topFoVRadians,
    float zNear, float zFar )
{
    GLFrustum frustum;

    frustum.left = zNear * tan( leftFovRadians );
    frustum.right = zNear * tan( rightFoVRadians );
    frustum.bottom = zNear * tan( bottomFoVRadians );
    frustum.top = zNear * tan( topFoVRadians );

    frustum.zNear = zNear;
    frustum.zFar = zFar;

    return frustum;
}

// static
GLFrustum GLFrustum::makeSymmetricPerspective(
    float fovYRadians, float aspectRatio,
    float zNear, float zFar )
{
    GLFrustum frustum;

    float tanHalfFovY = tan( 0.5f * fovYRadians );

    // tan( theta / 2 ) = top / zNear
    frustum.top = zNear * tanHalfFovY;
    frustum.bottom = -frustum.top;

    // aspectRatio = width / height
    //             = ( right - left ) / ( top - bottom )
    // But left = -right and bottom = -top:
    //             = (2 * right) / (2 * top)
    // --> right = aspectRatio * top.
    frustum.right = aspectRatio * frustum.top;
    frustum.left = -frustum.right;

    frustum.zNear = zNear;
    frustum.zFar = zFar;

    return frustum;
}

bool operator == ( const GLFrustum& f0, const GLFrustum& f1 )
{
    // If one of them is infinite but the other is not, they're not equal.
    if( isinf( f0.zFar ) ^ isinf( f1.zFar ) )
    {
        return false;
    }
    else
    {
        return( f0.left == f1.left &&
            f0.right == f1.right &&
            f0.bottom == f1.bottom &&
            f0.top == f1.top &&
            f0.zNear == f1.zNear );
    }
}

bool operator != ( const GLFrustum& f0, const GLFrustum& f1 )
{
    return !( f0 == f1 );
}

GLFrustum lerp( const GLFrustum& f0, const GLFrustum& f1, float t )
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

GLFrustum cubicInterpolate( const GLFrustum& f0, const GLFrustum& f1,
    const GLFrustum& f2, const GLFrustum& f3, float t )
{
    GLFrustum frustum;

    frustum.left = ::cubicInterpolate( f0.left, f1.left, f2.left, f3.left, t );
    frustum.right = ::cubicInterpolate( f0.right, f1.right, f2.right, f3.right, t );
    frustum.bottom = ::cubicInterpolate( f0.bottom, f1.bottom, f2.bottom, f3.bottom, t );
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

} } } // cameras, core, libcgt
