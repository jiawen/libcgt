#include "libcgt/core/io/BinaryFileInputStream.h"

BinaryFileInputStream::BinaryFileInputStream( const char* filename )
{
    m_fp = fopen( filename, "rb" );
}

// virtual
BinaryFileInputStream::~BinaryFileInputStream()
{
    close();
}

bool BinaryFileInputStream::isOpen() const
{
    return( m_fp != nullptr );
}

bool BinaryFileInputStream::close()
{
    if( m_fp != nullptr )
    {
        // fclose() returns 0 on success, EOF otherwise.
        int status = fclose( m_fp );
        m_fp = nullptr;
        return( status == 0 );
    }
    return false;
}
