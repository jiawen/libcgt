#pragma once

#include <vecmath/Rect2i.h>
#include <vecmath/Vector2f.h>

class CameraUtils
{
public:

	// note that y points up, for both the crop window
	// and the principal point
	static void adjustIntrinsicsToCrop
	(
		int imageWidth,
		int imageHeight,
		const Rect2i& cropWindow,

		const Vector2f& focalLengthPixels,
		const Vector2f& principalPointPixels,

		Vector2f& newFocalLengthPixels,
		Vector2f& newPrincipalPointPixels
	);

	// uniformScale is the parameter s such that x' = x * s
	// --> s = new_size / old_size
	// (when downsampling, s is < 1, when upsampling, s is > 1)
	static void adjustIntrinsicsToScale
	(
		int imageWidth,
		int imageHeight,
		
		float uniformScale,

		const Vector2f& focalLengthPixels,
		const Vector2f& principalPointPixels,

		Vector2f& newFocalLengthPixels,
		Vector2f& newPrincipalPointPixels
	);

	// for a centered perspective camera
	static float focalLengthPixelsToFoVRadians( float focalLengthPixels, float imageSize );
	static float fovRadiansToFocalLengthPixels( float fovRadians, float imageSize );

	// given potentially off-center intrinsics
	// (principal point != imageSize / 2,
	//  note that *y points up*, even for the principal point)
	// returns frustum parameters
	static void intrinsicsToFrustum
	(
		const Vector2f& focalLengthPixels, const Vector2f& principalPointPixels,
		const Vector2f& imageSize,
		
		float zNear,
		float& left, float& right,
		float& bottom, float& top
	);
};
