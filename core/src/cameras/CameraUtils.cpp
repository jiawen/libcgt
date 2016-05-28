#include "cameras/CameraUtils.h"

namespace libcgt { namespace core { namespace cameras {

Intrinsics adjustIntrinsicsToCrop( const Rect2i& cropWindow,
    Intrinsics input )
{
    // camera model:
    // x = fl * ( X / Z ) + pp
    // crop:
    // x' = x - left
    // x' = fl * ( X / Z ) + pp - left
    // ==>
    // fl' = fl
    // pp' = pp - left

    return
    {
        input.focalLength,
        input.principalPoint - cropWindow.origin()
    };
}

Intrinsics adjustIntrinsicsToScale( Intrinsics input,
    float uniformScale )
{
    // camera model:
    // x = fl * ( X / Z ) + pp
    // resample:
    // x' = uniformScale * x
    // x' = fl * ( X / Z ) * uniformScale + pp * uniformScale
    // ==>
    // fl' = fl * uniformScale
    // pp' = pp * uniformScale

    return
    {
        input.focalLength * uniformScale,
        input.principalPoint * uniformScale
    };
}

float focalLengthPixelsToFoVRadians( float focalLengthPixels, float imageSize )
{
    return 2 * atan( 0.5f * imageSize / focalLengthPixels );
}

Vector2f focalLengthPixelsToFoVRadians( const Vector2f& focalLengthPixels,
    const Vector2f& imageSize )
{
    return
    {
        focalLengthPixelsToFoVRadians( focalLengthPixels.x, imageSize.x ),
        focalLengthPixelsToFoVRadians( focalLengthPixels.y, imageSize.y )
    };
}

float fovRadiansToFocalLengthPixels( float fovRadians, float imageSize )
{
    return 0.5f * imageSize / tan( 0.5f * fovRadians );
}

Vector2f fovRadiansToFocalLengthPixels( const Vector2f& fovRadians,
    const Vector2f& imageSize )
{
    return
    {
        fovRadiansToFocalLengthPixels( fovRadians.x, imageSize.x ),
        fovRadiansToFocalLengthPixels( fovRadians.y, imageSize.y )
    };
}

GLFrustum intrinsicsToFrustum( const Intrinsics& input,
    const Vector2f& imageSize,
    float zNear, float zFar )
{
    // TODO: point to a diagram

    // pp.x / fl = -left / zNear
    // ( width - pp.x ) / fl = right / zNear
    GLFrustum output;

    output.left = -zNear * input.principalPoint.x / input.focalLength.x;
    output.right =
        zNear * ( imageSize.x - input.principalPoint.x ) / input.focalLength.x;

    output.bottom = -zNear * input.principalPoint.y / input.focalLength.y;
    output.top =
        zNear * ( imageSize.y - input.principalPoint.y ) / input.focalLength.y;

    output.zNear = zNear;
    output.zFar = zFar;

    return output;
}

Intrinsics frustumToIntrinsics( const GLFrustum& frustum,
    const Vector2f& imageSize )
{
    float fx = imageSize.x * frustum.zNear / ( frustum.right - frustum.left );
    float fy = imageSize.y * frustum.zNear / ( frustum.top - frustum.bottom );
    float cx = -frustum.left * fx / frustum.zNear;
    float cy = -frustum.bottom * fy / frustum.zNear;

    return
    {
        { fx, fy },
        { cx, cy }
    };
}

} } } // cameras, core, libcgt
