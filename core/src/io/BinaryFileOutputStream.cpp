#include "io/BinaryFileOutputStream.h"

BinaryFileOutputStream::BinaryFileOutputStream( const char* filename,
    bool append )
{
    if( append )
    {
        m_fp = fopen( filename, "ab" );
    }
    else
    {
        m_fp = fopen( filename, "wb" );
    }
}

BinaryFileOutputStream::BinaryFileOutputStream( BinaryFileOutputStream&& move )
{
    close();
    m_fp = move.m_fp;
    move.m_fp = nullptr;
}

BinaryFileOutputStream& BinaryFileOutputStream::operator = (
    BinaryFileOutputStream&& move )
{
    if( this != &move )
    {
        close();
        m_fp = move.m_fp;
        move.m_fp = nullptr;
    }
    return *this;
}

// virtual
BinaryFileOutputStream::~BinaryFileOutputStream()
{
    close();
}

bool BinaryFileOutputStream::isOpen() const
{
    return( m_fp != nullptr );
}

bool BinaryFileOutputStream::close()
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
