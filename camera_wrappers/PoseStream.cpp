#include "libcgt/camera_wrappers/PoseStream.h"

#include <cassert>
#include "libcgt/core/vecmath/Matrix3f.h"
#include "libcgt/core/vecmath/Vector3f.h"

namespace libcgt { namespace camera_wrappers {

const int FORMAT_VERSION = 1;

int frameSizeBytes( PoseStreamFormat format )
{
    switch( format )
    {
    case PoseStreamFormat::ROTATION_MATRIX_3X3_COL_MAJOR_AND_TRANSLATION_VECTOR_FLOAT:
        return 12 * sizeof( float );
    case PoseStreamFormat::MATRIX_4X4_COL_MAJOR_FLOAT:
        return 16 * sizeof( float );
    case PoseStreamFormat::ROTATION_QUATERNION_WXYZ_AND_TRANSLATION_VECTOR_FLOAT:
        return 7 * sizeof( float );
    case PoseStreamFormat::ROTATION_VECTOR_AND_TRANSLATION_VECTOR_FLOAT:
        return 6 * sizeof( float );
    default:
        return 0;
    }
}

PoseInputStream::PoseInputStream( const char* filename ) :
    m_stream( filename )
{
    bool ok = false;

    // Read header.
    char magic[ 5 ] = {};
    int version;
    ok = m_stream.read( magic[ 0 ] );
    ok = m_stream.read( magic[ 1 ] );
    ok = m_stream.read( magic[ 2 ] );
    ok = m_stream.read( magic[ 3 ] );
    ok = m_stream.read( version );

    if( ok && strcmp( magic, "pose" ) == 0 && version == FORMAT_VERSION )
    {
        m_stream.read( m_metadata );
        if( ok )
        {
            m_buffer.resize( frameSizeBytes( m_metadata.format ) );
        }
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

Array1DReadView< uint8_t > PoseInputStream::read( int& frameIndex,
    int64_t& timestamp )
{
    bool ok;
    if( isValid() )
    {
        ok = m_stream.read( frameIndex );
        if( ok )
        {
            ok = m_stream.read( timestamp );
            if( ok )
            {
                Array1DWriteView< uint8_t > wv = m_buffer;
                ok = m_stream.readArray( wv );
                if( ok )
                {
                    return m_buffer;
                }
            }
        }
    }

    return Array1DReadView< uint8_t >();
}

PoseOutputStream::PoseOutputStream( PoseStreamMetadata metadata,
    const char* filename ) :
    m_metadata( metadata )
{
    m_stream = BinaryFileOutputStream( filename );
    m_stream.write( 'p' );
    m_stream.write( 'o' );
    m_stream.write( 's' );
    m_stream.write( 'e' );
    m_stream.write< int >( FORMAT_VERSION );

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

bool PoseOutputStream::write( int frameIndex, int64_t timestamp,
    const Matrix3f& rotation, const Vector3f& translation )
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

} } // camera_wrappers, libcgt
