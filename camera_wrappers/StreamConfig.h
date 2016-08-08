#pragma once

#include <vecmath/Vector2i.h>

#include "PixelFormat.h"

namespace libcgt { namespace camera_wrappers {

// TODO(jiawen): why does this need constructors at all?
class StreamConfig
{
public:

#if 1
    StreamConfig() = default;
    StreamConfig( const Vector2i& _resolution, int _fps,
        PixelFormat _pixelFormat, bool _mirror ) :
        resolution( _resolution ),
        fps( _fps ),
        pixelFormat( _pixelFormat ),
        mirror( _mirror )
    {

    }
    StreamConfig( const StreamConfig& copy ) = default;
    StreamConfig& operator = ( const StreamConfig& copy ) = default;
#endif

    Vector2i resolution = Vector2i();
    int fps = 0;
    PixelFormat pixelFormat = PixelFormat::INVALID;
    // When false, it is a regular camera.
    // When true, it acts like a mirror / webcam.
    bool mirror = false;
};

} } // camera_wrappers, libcgt
