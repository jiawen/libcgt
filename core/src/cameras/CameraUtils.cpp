#include "cameras/CameraUtils.h"

// static
void CameraUtils::adjustIntrinsicsToCrop
(
	int imageWidth,
	int imageHeight,
	const Rect2i& cropWindow,

	const Vector2f& focalLengthPixels,
	const Vector2f& principalPointPixels,

	Vector2f& newFocalLengthPixels,
	Vector2f& newPrincipalPointPixels
)
{
	// camera model:
	// x = fl * ( X / Z ) + pp
	// crop:
	// x' = x - left
	// x' = fl * ( X / Z ) + pp - left
	// ==>
	// fl' = fl
	// pp' = pp - left

    // image size is ignored
    ( void )imageWidth;
    ( void )imageHeight;

	newFocalLengthPixels = focalLengthPixels;
	newPrincipalPointPixels = principalPointPixels - cropWindow.origin();
}

// static
void CameraUtils::adjustIntrinsicsToScale
(
	int imageWidth,
	int imageHeight,

	float uniformScale,

	const Vector2f& focalLengthPixels,
	const Vector2f& principalPointPixels,

	Vector2f& newFocalLengthPixels,
	Vector2f& newPrincipalPointPixels
)
{
	// camera model:
	// x = fl * ( X / Z ) + pp
	// resample:
	// x' = uniformScale * x
	// x' = fl * ( X / Z ) * uniformScale + pp * uniformScale
	// ==>
	// fl' = fl * uniformScale
	// pp' = pp * uniformScale

	// image size is ignored
	(void)imageWidth;
	(void)imageHeight;

	newFocalLengthPixels = focalLengthPixels * uniformScale;
	newPrincipalPointPixels = principalPointPixels * uniformScale;
}

// static
float CameraUtils::focalLengthPixelsToFoVRadians( float focalLengthPixels, float imageSize )
{
	return 2 * atan( 0.5f * imageSize / focalLengthPixels );
}

// static
float CameraUtils::fovRadiansToFocalLengthPixels( float fovRadians, float imageSize )
{
	return 0.5f * imageSize / tan( 0.5f * fovRadians );
}

// static
void CameraUtils::intrinsicsToFrustum( const Vector2f& focalLengthPixels, const Vector2f& principalPointPixels,
	const Vector2f& imageSize,
	
	float zNear,
	float& left, float& right,
	float& bottom, float& top )
{
	// TODO: point to a diagram

	// pp.x / fl = -left / zNear
	// ( width - pp.x ) / fl = right / zNear

	left = -zNear * principalPointPixels.x / focalLengthPixels.x;
	right = zNear * ( imageSize.x - principalPointPixels.x ) / focalLengthPixels.x;

	bottom = -zNear * principalPointPixels.y / focalLengthPixels.y;
	top = zNear * ( imageSize.y - principalPointPixels.y ) / focalLengthPixels.y;
}
