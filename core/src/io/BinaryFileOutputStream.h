#pragma once

#include <cstdio>

#include <common/Array1DView.h>

class BinaryFileOutputStream
{
public:

    BinaryFileOutputStream() = default;
    BinaryFileOutputStream( const char* filename, bool append = false );
    virtual ~BinaryFileOutputStream();

    BinaryFileOutputStream( const BinaryFileOutputStream& copy ) = delete;
    BinaryFileOutputStream( BinaryFileOutputStream&& move );
    BinaryFileOutputStream& operator = (
        const BinaryFileOutputStream& copy ) = delete;

    // TODO(VS2015): default
    BinaryFileOutputStream& operator = ( BinaryFileOutputStream&& move );

    // Returns true if the file was properly opened.
    bool isOpen() const;

    // Returns true if the file was properly closed.
    bool close();

    // Flush the last write operation.
    bool flush() const;

    template< typename T >
    bool write( const T& x ) const;

    template< typename T >
    bool writeArray( Array1DView< const T > data ) const;

private:

    FILE* m_fp = nullptr;
};

template< typename T >
bool BinaryFileOutputStream::write( const T& x ) const
{
    size_t itemsWritten = fwrite( &x, sizeof( T ), 1, m_fp );
    return( itemsWritten == 1 );
}

template< typename T >
bool BinaryFileOutputStream::writeArray( Array1DView< const T > data ) const
{
    if( data.packed() )
    {
        size_t itemsWritten = fwrite(
            data.pointer(), sizeof( T ), data.size(), m_fp );
        return( itemsWritten == data.size() );
    }
    else
    {
        for( size_t i = 0; i < data.size(); ++i )
        {
            bool b = write< T >( data[ i ] );
            if( !b )
            {
                return b;
            }
        }
        return true;
    }
}
