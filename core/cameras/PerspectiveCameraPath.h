#pragma once

#include <vector>

#include "libcgt/core/cameras/PerspectiveCamera.h"

class PerspectiveCameraPath
{
public:

    PerspectiveCameraPath();

    // add a keyframe
    // keyframes are always 1 second apart
    void addKeyframe( const PerspectiveCamera& camera );

    int numKeyFrames();

    // clears all keyframes
    void clear();

    // removes the last keyframe
    void removeLastKeyframe();

    // gets the camera at a time t
    // t is in seconds
    PerspectiveCamera getCamera( float t );

    void load( const char* filename );
    void save( const char* filename );

private:

    std::vector< PerspectiveCamera > m_keyFrames;

};
