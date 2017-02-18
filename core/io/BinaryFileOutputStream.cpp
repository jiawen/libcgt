#include "libcgt/core/io/BinaryFileOutputStream.h"

BinaryFileOutputStream::BinaryFileOutputStream( const std::string& filename,
    bool append )
{
    if( append )
    {
        m_fp = fopen( filename.c_str(), "ab" );
    }
    else
    {
        m_fp = fopen( filename.c_str(), "wb" );
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

bool BinaryFileOutputStream::flush() const
{
    if( m_fp != nullptr )
    {
        // fflush() returns 0 on success, EOF otherwise.
        int status = fflush( m_fp );
        return( status == 0 );
    }
    return false;
}
