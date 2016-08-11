#include "RGBDStream.h"

#include <cassert>

namespace libcgt { namespace camera_wrappers {

const int FORMAT_VERSION = 1;

RGBDInputStream::RGBDInputStream( const char* filename ) :
    m_stream( filename )
{
    bool ok;

    // Read header.
    char magic[ 5 ] = {};
    int version;
    ok = m_stream.read( magic[ 0 ] );
    ok = m_stream.read( magic[ 1 ] );
    ok = m_stream.read( magic[ 2 ] );
    ok = m_stream.read( magic[ 3 ] );
    ok = m_stream.read( version );

    if( ok && strcmp( magic, "rgbd" ) == 0 && version == FORMAT_VERSION )
    {
        int nStreams = 0;
        ok = m_stream.read( nStreams );
        if( ok && nStreams > 0 )
        {
            m_metadata.resize( nStreams );
            for( int i = 0; i < nStreams; ++i )
            {
                ok = m_stream.read( m_metadata[ i ] );
                if( ok )
                {
                    int bufferSize =
                        pixelSizeBytes( m_metadata[ i ].format ) *
                        m_metadata[ i ].size.x * m_metadata[ i ].size.y;
                    m_buffers.emplace_back( bufferSize );
                }

                if( !ok )
                {
                    m_metadata.clear();
                    break;
                }
            }
        }
    }

    m_valid = ok;
}

bool RGBDInputStream::isValid() const
{
    return m_valid && m_stream.isOpen();
}

const std::vector< StreamMetadata>& RGBDInputStream::metadata() const
{
    return m_metadata;
}

Array1DView< const uint8_t > RGBDInputStream::read( uint32_t& streamId,
    int& frameId, int64_t& timestamp )
{
    bool ok;
    if( isValid() )
    {
        ok = m_stream.read( streamId );
        if( ok && streamId >= 0 && streamId < m_buffers.size() )
        {
            ok = m_stream.read( frameId );
            if( ok )
            {
                ok = m_stream.read( timestamp );
                if( ok )
                {
                    ok = m_stream.readArray(
                        m_buffers[ streamId ].writeView() );
                    if( ok )
                    {
                        return m_buffers[ streamId ].readView();
                    }
                }
            }
        }
    }

    return Array1DView< const uint8_t >();
}

RGBDOutputStream::RGBDOutputStream(
    const std::vector< StreamMetadata >& metadata,
    const char* filename ) :
    m_metadata( metadata )
{
    int nStreams = static_cast< int >( metadata.size() );
    assert( nStreams > 0 );

    if( nStreams > 0 )
    {
        m_stream = BinaryFileOutputStream( filename );
        m_stream.write( 'r' );
        m_stream.write( 'g' );
        m_stream.write( 'b' );
        m_stream.write( 'd' );
        m_stream.write< int >( FORMAT_VERSION );

        m_stream.write( nStreams );

        for( size_t i = 0; i < metadata.size(); ++i )
        {
            m_stream.write( metadata[ i ] );
        }
    }
}

// virtual
RGBDOutputStream::~RGBDOutputStream()
{
    close();
}

RGBDOutputStream::RGBDOutputStream( RGBDOutputStream&& move )
{
    close();
    m_stream = std::move( move.m_stream );
    m_metadata = std::move( move.m_metadata );

    move.m_stream = BinaryFileOutputStream();
    move.m_metadata = std::vector< StreamMetadata >();
}

RGBDOutputStream& RGBDOutputStream::operator = ( RGBDOutputStream&& move )
{
    if( this != &move )
    {
        close();
        m_stream = std::move( move.m_stream );
        m_metadata = std::move( move.m_metadata );

        move.m_stream = BinaryFileOutputStream();
        move.m_metadata = std::vector< StreamMetadata >();
    }
    return *this;
}

bool RGBDOutputStream::isValid() const
{
    return m_stream.isOpen();
}

bool RGBDOutputStream::close()
{
    return m_stream.close();
}

bool RGBDOutputStream::write( uint32_t streamId, int frameId,
    int64_t timestamp, Array1DView< const uint8_t > data ) const
{
    if( streamId >= m_metadata.size() )
    {
        return false;
    }

    if( !m_stream.write( streamId ) )
    {
        return false;
    }

    if( !m_stream.write( frameId ) )
    {
        return false;
    }

    if( !m_stream.write( timestamp ) )
    {
        return false;
    }

    return m_stream.writeArray( data );
}

} } // camera_wrappers, libcgt
