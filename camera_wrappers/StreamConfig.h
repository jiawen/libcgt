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
    StreamConfig( const Vector2i& _resolution, int _fps, PixelFormat _pixelFormat ) :
        resolution( _resolution ),
        fps( _fps ),
        pixelFormat( _pixelFormat )
    {

    }
    StreamConfig( const StreamConfig& copy ) = default;
    StreamConfig& operator = ( const StreamConfig& copy ) = default;
#endif

    Vector2i resolution = Vector2i();
    int fps = 0;
    PixelFormat pixelFormat = PixelFormat::INVALID;
};

} } // camera_wrappers, libcgt
