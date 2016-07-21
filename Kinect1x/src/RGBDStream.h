#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <common/Array1D.h>
#include <common/Array2D.h>
#include <io/BinaryFileInputStream.h>
#include <io/BinaryFileOutputStream.h>
#include <vecmath/Vector2i.h>

struct StreamMetadata
{
    enum class Format : uint32_t
    {
        DEPTH_MM_UINT16 = 0,
        DEPTH_M_FLOAT32 = 1,
        RGBA_UINT8 = 2,
        RGB_UINT8 = 3,
        BGRA_UINT8 = 4,
        BGR_UINT8 = 5
    };

    Format format;
    uint32_t elementSizeBytes;
    Vector2i size; // width, height
};

class RGBDInputStream
{
public:

    RGBDInputStream( const char* filename );

    RGBDInputStream( const RGBDInputStream& copy ) = delete;
    RGBDInputStream& operator = ( const RGBDInputStream& copy ) = delete;

    bool isValid() const;

    const std::vector< StreamMetadata>& metadata() const;

    Array1DView< const uint8_t > read( int& streamId, int& frameId,
        int64_t& timestamp );

private:

    BinaryFileInputStream m_stream;
    std::vector< StreamMetadata > m_metadata;
    std::vector< Array1D< uint8_t > > m_buffers;
    bool m_valid;

};

class RGBDOutputStream
{
public:

    RGBDOutputStream( const std::vector< StreamMetadata >& metadata,
        const char* filename );
    virtual ~RGBDOutputStream();

    RGBDOutputStream( const RGBDOutputStream& copy ) = delete;
    RGBDOutputStream& operator = ( const RGBDOutputStream& copy ) = delete;

    bool close();

    // TODO(jiawen): check that frameId and timestamp is monotonically
    // increasing.
    bool writeFrame( uint32_t streamId, int frameId, int64_t timestamp,
        Array1DView< const uint8_t > data ) const;

private:

    BinaryFileOutputStream m_stream;
    std::vector< StreamMetadata > m_metadata;
};
