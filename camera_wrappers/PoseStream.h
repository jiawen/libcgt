#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <common/Array1D.h>
#include <common/Array2D.h>
#include <io/BinaryFileInputStream.h>
#include <io/BinaryFileOutputStream.h>
#include <vecmath/Vector2i.h>

#include "libcgt/camera_wrappers/PixelFormat.h"

class Matrix3f;
class Vector3f;

namespace libcgt { namespace camera_wrappers {

enum class PoseStreamFormat : uint32_t
{
    INVALID = 0,

    // 12 floats: 3 columns of the rotation matrix then 3 for translation.
    ROTATION_MATRIX_3X3_COL_MAJOR_AND_TRANSLATION_VECTOR_FLOAT = 1,

    // 16 floats.
    MATRIX_4X4_COL_MAJOR_FLOAT = 2,

    // 7 floats: wxyz of the quaternion, then 3 for translation.
    ROTATION_QUATERNION_WXYZ_AND_TRANSLATION_VECTOR_FLOAT = 3,

    // 6 floats: xyz for rotation, then 3 for translation.
    ROTATION_VECTOR_AND_TRANSLATION_VECTOR_FLOAT = 4
};

enum class PoseStreamTransformDirection : uint32_t
{
    WORLD_FROM_CAMERA = 0,
    CAMERA_FROM_WORLD = 1
};

struct PoseStreamMetadata
{
    PoseStreamFormat format = PoseStreamFormat::INVALID;
    PoseStreamTransformDirection direction =
        PoseStreamTransformDirection::WORLD_FROM_CAMERA;
};

class PoseInputStream
{
public:

    PoseInputStream( const char* filename );

    PoseInputStream( const PoseInputStream& copy ) = delete;
    PoseInputStream& operator = ( const PoseInputStream& copy ) = delete;

    bool isValid() const;

    const PoseStreamMetadata& metadata() const;

    Array1DReadView< uint8_t > read( int& frameIndex, int64_t& timestamp );

private:

    BinaryFileInputStream m_stream;
    PoseStreamMetadata m_metadata;
    Array1D< uint8_t > m_buffer;
    bool m_valid;

};

class PoseOutputStream
{
public:

    PoseOutputStream() = default;
    PoseOutputStream( PoseStreamMetadata metadata, const char* filename );
    virtual ~PoseOutputStream();

    PoseOutputStream( const PoseOutputStream& copy ) = delete;
    PoseOutputStream& operator = ( const PoseOutputStream& copy ) = delete;
    PoseOutputStream( PoseOutputStream&& move );
    PoseOutputStream& operator = ( PoseOutputStream&& move );

    bool isValid() const;

    bool close();

    // TODO(jiawen): check that frameIndex and timestamp is monotonically
    // increasing.
    bool write( int frameIndex, int64_t timestamp,
        const Matrix3f& rotation, const Vector3f& translation );

private:

    BinaryFileOutputStream m_stream;
    PoseStreamMetadata m_metadata;
};

} } // camera_wrappers, libcgt
