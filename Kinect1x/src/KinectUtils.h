#pragma once

#include <windows.h>
#include <NuiApi.h>
#include <cstdint>

#include <common/BasicTypes.h>
#include <common/Array2DView.h>
#include <vecmath/Matrix4f.h>

class KinectUtils
{
public:

    // Given a frame (in particular, its estimate of the ground plane)
    // Returns a matrix mapping points in the Kinect's frame (such as skeleton joints)
    // to points in a virtual world where:
    //
    // the x-axis is the same as the Kinect's
    // the y-axis is the ground
    // (the z-axis is the x cross y)
    // and the point on the ground plane closest to the Kinect is the origin
    static Matrix4f kinectToWorld( const NUI_SKELETON_FRAME& frame );
    static Matrix4f worldToKinect( const NUI_SKELETON_FRAME& frame );

    // Given the raw depth map in 16-bit shorts,
    // converts it into a floating point image in meters
    //
    // Options:
    //
    // flipLeftRight (default = true)
    //   By default, the Kinect treats the camera as a mirror (webcam looking at the user),
    //     flipping the image left <--> right
    //   we flip it again when using it as a camera.
    //
    // rightShift (default = 0)
    //   When a Kinect for Windows sensor is initialized with NUI_INITIALIZE_FLAG_USES_DEPTH
    //     each ushort contains the depth in millimeters, in the 12 bits d[11:0].
    //   However, when a Kinect for Windows sensor is initialized with NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX
    //     each ushort contains the depth in millimeters, in the 12 bits d[15:3].
    //     and the player index on the bottom 3 bits d[2:0]
    //
    //
    static void rawDepthMapToMeters( Array2DView< const uint16_t > rawDepth,
        Array2DView< float > outputMeters, bool flipLeftRight = true,
        int rightShft = 0 );

    // Convert a NUI_IMAGE_RESOLUTION to a numerical resolution in pixels.
    // Returns (0, 0) on NUI_IMAGE_RESOLUTION_INVALID.
    static Vector2i toVector2i( NUI_IMAGE_RESOLUTION resolution );

};
