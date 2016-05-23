#pragma once

#include <limits>

#include <vecmath/Rect2i.h>
#include <vecmath/Vector2f.h>
#include "cameras/GLFrustum.h"
#include "cameras/Intrinsics.h"

namespace libcgt { namespace core { namespace cameras {

// Computes new camera intrinsics when an image is cropped.
// It is independent of original image size.
//
// This has only been tested when y points up for both the crop window and the
// the principal point.
// TODO(jiawen): verify that this works when y points down.
Intrinsics adjustIntrinsicsToCrop( const Intrinsics& input,
    const Rect2i& cropWindow );

// Computes new camera intrinsics when an image is uniformly scaled.
// It is independent of original image size.
//
// uniformScale is the parameter s such that x' = x * s
// --> s = new_size / old_size
// (when downsampling, s is < 1, when upsampling, s is > 1)
Intrinsics adjustIntrinsicsToScale( const Intrinsics& input,
    float uniformScale );

// For a *centered* perspective camera, convert focal length (in pixels) to
// field of view (in radians).
float focalLengthPixelsToFoVRadians( float focalLengthPixels,
    float imageSize );
Vector2f focalLengthPixelsToFoVRadians( const Vector2f& focalLengthPixels,
    const Vector2f& imageSize );

// For a *centered* perspective camera, convert field of view (in radians) to
// focal length (in pixels).
float fovRadiansToFocalLengthPixels( float fovRadians, float imageSize );
Vector2f fovRadiansToFocalLengthPixels( const Vector2f& fovRadians,
    const Vector2f& imageSize);

// Convert camera intrinsics to an OpenGL frustum.
//
// The intrinsics *must* be given in the OpenGL convention, such that the
// principal point is in a coordinate system where *y points up*.
//
// The principal point does not have to be centered (it does not have to equal
// half the image size in both both dimensions).
GLFrustum intrinsicsToFrustum( const Intrinsics& input,
    const Vector2f& imageSize,
    float zNear, float zFar = std::numeric_limits< float >::infinity() );

} } } // cameras, core, libcgt
