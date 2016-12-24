#pragma once

#include "libcgt/core/vecmath/Vector2i.h"

#include "libcgt/camera_wrappers/PixelFormat.h"
#include "libcgt/camera_wrappers/StreamType.h"

namespace libcgt { namespace camera_wrappers {

// TODO(jiawen): why does this need constructors at all?
class StreamConfig
{
public:

#if 1
    StreamConfig() = default;
    StreamConfig( StreamType _type, const Vector2i& _resolution,
        PixelFormat _pixelFormat, int _fps, bool _mirror ) :
        type( _type ),
        resolution( _resolution ),
        fps( _fps ),
        pixelFormat( _pixelFormat ),
        mirror( _mirror )
    {

    }
    StreamConfig( const StreamConfig& copy ) = default;
    StreamConfig& operator = ( const StreamConfig& copy ) = default;
#endif

    StreamType type = StreamType::UNKNOWN;
    Vector2i resolution = Vector2i();
    PixelFormat pixelFormat = PixelFormat::INVALID;

    int fps = 0;

    // When false, it is a regular camera.
    // When true, it acts like a mirror / webcam.
    bool mirror = false;
};

} } // camera_wrappers, libcgt
