#include "RGBDStream.h"

#include <cassert>

RGBDInputStream::RGBDInputStream( const char* filename ) :
    m_stream( filename )
{
    bool ok;

    // Read header.
    char magic[ 5 ] = {};
    ok = m_stream.read( magic[ 0 ] );
    ok = m_stream.read( magic[ 1 ] );
    ok = m_stream.read( magic[ 2 ] );
    ok = m_stream.read( magic[ 3 ] );

    if( ok && strcmp( magic, "rgbd" ) == 0 )
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
                    int bufferSize = m_metadata[ i ].elementSizeBytes *
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

Array1DView< const uint8_t > RGBDInputStream::read( int& streamId,
    int& frameId, int64_t& timestamp )
{
    bool ok;
    if( isValid() )
    {
        ok = m_stream.read( streamId );
        if( ok && streamId >= 0 && streamId < m_buffers.size() )
        {
            m_stream.read( frameId );
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

bool RGBDOutputStream::close()
{
    return m_stream.close();
}

bool RGBDOutputStream::writeFrame( uint32_t streamId, int frameId,
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
