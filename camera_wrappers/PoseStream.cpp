#include "libcgt/camera_wrappers/PoseStream.h"

#include <cassert>
#include "libcgt/core/vecmath/Matrix3f.h"
#include "libcgt/core/vecmath/Matrix4f.h"
#include "libcgt/core/vecmath/Quat4f.h"
#include "libcgt/core/vecmath/Vector3f.h"

namespace
{

size_t frameSizeBytes( libcgt::camera_wrappers::PoseStreamFormat format )
{
    switch( format )
    {
    case libcgt::camera_wrappers::PoseStreamFormat::ROTATION_MATRIX_3X3_COL_MAJOR_AND_TRANSLATION_VECTOR_FLOAT:
        return 12 * sizeof( float );
    case libcgt::camera_wrappers::PoseStreamFormat::MATRIX_4X4_COL_MAJOR_FLOAT:
        return 16 * sizeof( float );
    case libcgt::camera_wrappers::PoseStreamFormat::ROTATION_QUATERNION_WXYZ_AND_TRANSLATION_VECTOR_FLOAT:
        return 7 * sizeof( float );
    case libcgt::camera_wrappers::PoseStreamFormat::ROTATION_VECTOR_AND_TRANSLATION_VECTOR_FLOAT:
        return 6 * sizeof( float );
    default:
        return 0;
    }
}

}

namespace libcgt { namespace camera_wrappers {

const int32_t FORMAT_VERSION = 1;

PoseInputStream::PoseInputStream( const std::string& filename ) :
    m_stream( filename )
{
    bool ok = false;

    // Read header.
    char magic[ 5 ] = {};
    int32_t version;
    ok = m_stream.read( magic[ 0 ] );
    ok = m_stream.read( magic[ 1 ] );
    ok = m_stream.read( magic[ 2 ] );
    ok = m_stream.read( magic[ 3 ] );
    ok = m_stream.read( version );

    if( ok && strcmp( magic, "pose" ) == 0 && version == FORMAT_VERSION )
    {
        ok = m_stream.read( m_metadata );
    }

    m_valid = ok;
}

bool PoseInputStream::isValid() const
{
    return m_valid && m_stream.isOpen();
}

const PoseStreamMetadata& PoseInputStream::metadata() const
{
    return m_metadata;
}

template< typename RotationType, typename TranslationType >
bool PoseInputStream::read( int32_t& frameIndex, int64_t& timestamp,
    RotationType& rotation, TranslationType& translation )
{
    bool ok = isValid() &&
        (
            ( m_metadata.format == PoseStreamFormat::ROTATION_MATRIX_3X3_COL_MAJOR_AND_TRANSLATION_VECTOR_FLOAT ) ||
            ( m_metadata.format == PoseStreamFormat::ROTATION_QUATERNION_WXYZ_AND_TRANSLATION_VECTOR_FLOAT ) ||
            ( m_metadata.format == PoseStreamFormat::ROTATION_VECTOR_AND_TRANSLATION_VECTOR_FLOAT )
        );
    if( ok )
    {
        ok = m_stream.read( frameIndex );
        if( ok )
        {
            ok = m_stream.read( timestamp );
            if( ok )
            {
                if( ok )
                {
                    ok = m_stream.read( rotation );
                    if( ok )
                    {
                        ok = m_stream.read( translation );
                    }
                }
            }
        }
    }

    return ok;
}

// Explicitly instantiate template.
template bool PoseInputStream::read( int32_t&, int64_t&,
    Matrix3f&, Vector3f& );
template bool PoseInputStream::read( int32_t&, int64_t&, Quat4f&, Vector3f& );
template bool PoseInputStream::read( int32_t&, int64_t&,
    Vector3f&, Vector3f& );

bool PoseInputStream::read( int32_t& frameIndex, int64_t& timestamp,
    Matrix4f& pose )
{
    bool ok = isValid() &&
        m_metadata.format == PoseStreamFormat::MATRIX_4X4_COL_MAJOR_FLOAT;
    if( ok )
    {
        ok = m_stream.read( frameIndex );
        if( ok )
        {
            ok = m_stream.read( timestamp );
            if( ok )
            {
                ok = m_stream.read( pose );
            }
        }
    }

    return ok;
}

PoseOutputStream::PoseOutputStream( PoseStreamMetadata metadata,
    const std::string& filename ) :
    m_metadata( metadata )
{
    m_stream = BinaryFileOutputStream( filename );
    m_stream.write( 'p' );
    m_stream.write( 'o' );
    m_stream.write( 's' );
    m_stream.write( 'e' );
    m_stream.write< int32_t >( FORMAT_VERSION );

    m_stream.write( metadata );
}

// virtual
PoseOutputStream::~PoseOutputStream()
{
    close();
}

PoseOutputStream::PoseOutputStream( PoseOutputStream&& move )
{
    close();
    m_stream = std::move( move.m_stream );
    m_metadata = std::move( move.m_metadata );

    move.m_stream = BinaryFileOutputStream();
    move.m_metadata = PoseStreamMetadata();
}

PoseOutputStream& PoseOutputStream::operator = ( PoseOutputStream&& move )
{
    if( this != &move )
    {
        close();
        m_stream = std::move( move.m_stream );
        m_metadata = std::move( move.m_metadata );

        move.m_stream = BinaryFileOutputStream();
        move.m_metadata = PoseStreamMetadata();
    }
    return *this;
}

bool PoseOutputStream::isValid() const
{
    return m_stream.isOpen();
}

bool PoseOutputStream::close()
{
    return m_stream.close();
}

template< typename RotationType, typename TranslationType >
bool PoseOutputStream::write( int32_t frameIndex, int64_t timestamp,
    const RotationType& rotation, const TranslationType& translation )
{
    if( !m_stream.write( frameIndex ) )
    {
        return false;
    }

    if( !m_stream.write( timestamp ) )
    {
        return false;
    }

    if( !m_stream.write( rotation ) )
    {
        return false;
    }

    if( !m_stream.write( translation ) )
    {
        return false;
    }

    return true;
}

// Explicitly instantiate template.
template bool PoseOutputStream::write( int32_t, int64_t,
    const Matrix3f&, const Vector3f& );
template bool PoseOutputStream::write( int32_t, int64_t,
    const Quat4f&, const Vector3f& );
template bool PoseOutputStream::write( int32_t, int64_t,
    const Vector3f&, const Vector3f& );

bool PoseOutputStream::write( int32_t frameIndex, int64_t timestamp,
    const Matrix4f& pose )
{
    if( !m_stream.write( frameIndex ) )
    {
        return false;
    }

    if( !m_stream.write( timestamp ) )
    {
        return false;
    }

    if( !m_stream.write( pose ) )
    {
        return false;
    }

    return true;
}

} } // camera_wrappers, libcgt
