#pragma once

#include <cstdio>
#include <string>

#include "libcgt/core/common/ArrayView.h"

class BinaryFileInputStream
{
public:

    BinaryFileInputStream( const std::string& filename );
    virtual ~BinaryFileInputStream();

    BinaryFileInputStream( const BinaryFileInputStream& copy ) = delete;
    BinaryFileInputStream& operator =
        ( const BinaryFileInputStream& copy ) = delete;

    // Returns true if the file was properly opened.
    bool isOpen() const;

    // Returns true if the stream was properly closed.
    bool close();

    // T must be a primitive type or a struct without pointer members.
    // Returns false on error or once end of file is reached.
    template< typename T >
    bool read( T& output );

    // T must be a primitive type or a struct without pointer members.
    // Returns false on error or once end of file is reached.
    template< typename T >
    bool readArray( Array1DWriteView< T > output );

private:

    FILE* m_fp = nullptr;
};

template< typename T >
bool BinaryFileInputStream::read( T& output )
{
    if( isOpen() )
    {
        size_t itemsRead = fread( &output, sizeof( T ), 1, m_fp );
        return( itemsRead == 1 );
    }
    else
    {
        return false;
    }
}

template< typename T >
bool BinaryFileInputStream::readArray( Array1DWriteView< T > output )
{
    if( output.isNull() || !output.packed() )
    {
        return false;
    }

    if( isOpen() )
    {
        size_t itemsRead = fread( output.pointer(), sizeof( T ), output.size(),
            m_fp );
        return( itemsRead == output.size() );
    }
    else
    {
        return false;
    }
}
