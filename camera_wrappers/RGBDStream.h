#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "libcgt/core/common/Array1D.h"
#include "libcgt/core/common/Array2D.h"
#include "libcgt/core/io/BinaryFileInputStream.h"
#include "libcgt/core/io/BinaryFileOutputStream.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/camera_wrappers/PixelFormat.h"
#include "libcgt/camera_wrappers/StreamType.h"

namespace libcgt { namespace camera_wrappers {

struct StreamMetadata
{
    StreamType type;
    PixelFormat format;
    Vector2i size; // width, height
};

class RGBDInputStream
{
public:

    RGBDInputStream( const std::string& filename );

    RGBDInputStream( const RGBDInputStream& copy ) = delete;
    RGBDInputStream& operator = ( const RGBDInputStream& copy ) = delete;

    bool isValid() const;

    const std::vector< StreamMetadata >& metadata() const;

    Array1DReadView< uint8_t > read( uint32_t& streamId,
        int32_t& frameIndex, int64_t& timestamp );

private:

    BinaryFileInputStream m_stream;
    std::vector< StreamMetadata > m_metadata;
    std::vector< Array1D< uint8_t > > m_buffers;
    bool m_valid;

};

class RGBDOutputStream
{
public:

    RGBDOutputStream() = default;
    RGBDOutputStream( const std::vector< StreamMetadata >& metadata,
        const std::string& filename );
    virtual ~RGBDOutputStream();

    RGBDOutputStream( const RGBDOutputStream& copy ) = delete;
    RGBDOutputStream& operator = ( const RGBDOutputStream& copy ) = delete;
    RGBDOutputStream( RGBDOutputStream&& move );
    RGBDOutputStream& operator = ( RGBDOutputStream&& move );

    bool isValid() const;

    bool close();

    // TODO(jiawen): check that frameIndex and timestamp is monotonically
    // increasing.
    bool write( uint32_t streamId, int32_t frameIndex, int64_t timestamp,
        Array1DReadView< uint8_t > data ) const;

private:

    BinaryFileOutputStream m_stream;
    std::vector< StreamMetadata > m_metadata;
};

} } // camera_wrappers, libcgt
