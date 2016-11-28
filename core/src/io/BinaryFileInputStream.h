#pragma once

#include <cstdio>

#include <common/ArrayView.h>

class BinaryFileInputStream
{
public:

    BinaryFileInputStream( const char* filename );
    virtual ~BinaryFileInputStream();

    BinaryFileInputStream( const BinaryFileInputStream& copy ) = delete;
    BinaryFileInputStream& operator =
        ( const BinaryFileInputStream& copy ) = delete;

    // Returns true if the file was properly opened.
    bool isOpen() const;

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
    size_t itemsRead = fread( &output, sizeof( T ), 1, m_fp );
    return( itemsRead == 1 );
}

template< typename T >
bool BinaryFileInputStream::readArray( Array1DWriteView< T > output )
{
    if( output.isNull() || !output.packed() )
    {
        return false;
    }

    size_t itemsRead = fread( output.pointer(), sizeof( T ), output.size(),
        m_fp );
    return( itemsRead == output.size() );
}
