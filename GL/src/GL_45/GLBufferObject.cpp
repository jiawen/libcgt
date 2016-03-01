#include <cassert>
#include <common/ArrayUtils.h>

#include "GLBufferObject.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
bool GLBufferObject::copy( GLBufferObject* pSource,
    GLBufferObject* pDestination, GLintptr sourceOffsetBytes,
    GLintptr destinationOffsetBytes, GLsizeiptr nBytes )
{
    if( sourceOffsetBytes >= pSource->numBytes() ||
        destinationOffsetBytes >= pDestination->numBytes() ||
        sourceOffsetBytes + nBytes >= pSource->numBytes() ||
        destinationOffsetBytes + nBytes >= pDestination->numBytes() )
    {
        return false;
    }

    glCopyNamedBufferSubData( pSource->m_id, pDestination->m_id,
        sourceOffsetBytes, destinationOffsetBytes,
        nBytes );
    return true;
}

GLBufferObject::GLBufferObject( GLsizeiptr nBytes,
    GLBufferObject::StorageFlags flags ) :
    m_id( 0 ),
    m_nBytes( nBytes ),
    m_storageFlags( flags )
{
    glCreateBuffers( 1, &m_id );
    glNamedBufferStorage( m_id, nBytes, nullptr, glStorageFlags() );
}

GLBufferObject::GLBufferObject( Array1DView< uint8_t > data,
    GLBufferObject::StorageFlags flags ) :
    m_id( 0 ),
    m_nBytes( data.size() ),
    m_storageFlags( flags )
{
    assert( data.notNull() && data.packed() );
    glCreateBuffers( 1, &m_id );
    glNamedBufferStorage( m_id, m_nBytes, data.pointer(), glStorageFlags() );
}

// virtual
GLBufferObject::~GLBufferObject()
{
    glDeleteBuffers( 1, &m_id );
}

GLuint GLBufferObject::id() const
{
    return m_id;
}

size_t GLBufferObject::numBytes() const
{
    return m_nBytes;
}

GLBufferObject::StorageFlags GLBufferObject::storageFlags() const
{
    return m_storageFlags;
}

GLbitfield GLBufferObject::glStorageFlags() const
{
    return ::glStorageFlags( m_storageFlags );
}

Array1DView< uint8_t > GLBufferObject::mapRange( GLintptr offsetBytes,
    GLsizeiptr sizeBytes, GLBufferObject::MapRangeAccess access )
{
    return Array1DView< uint8_t >(
        glMapNamedBufferRange( m_id, offsetBytes, sizeBytes,
            glBufferMapRangeAccess( access ) ),
        sizeBytes
    );
}

void GLBufferObject::flushRange( GLintptr offsetBytes, GLintptr sizeBytes )
{
    glFlushMappedNamedBufferRange( m_id, offsetBytes, sizeBytes );
}

void GLBufferObject::invalidate()
{
    glInvalidateBufferData( m_id );
}

void GLBufferObject::invalidateRange( GLintptr offset, GLintptr size )
{
    glInvalidateBufferSubData( m_id, offset, size );
}

void GLBufferObject::unmap()
{
    glUnmapNamedBuffer( m_id );
}

bool GLBufferObject::get( GLintptr srcOffset, Array1DView< uint8_t > dst )
{
    if( srcOffset + dst.size() > m_nBytes )
    {
        return false;
    }

    glGetNamedBufferSubData( m_id, srcOffset, dst.size(), dst.pointer() );
    return true;
}

bool GLBufferObject::set( Array1DView< const uint8_t > src, GLintptr dstOffset )
{
    if( !( src.packed() ) )
    {
        return false;
    }
    if( dstOffset + src.size() > m_nBytes )
    {
        return false;
    }

    glNamedBufferSubData( m_id, dstOffset, src.size(), src.pointer() );
    return true;
}

void GLBufferObject::clear()
{
    glClearNamedBufferData( m_id, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL );
}

void GLBufferObject::clear( GLImageInternalFormat dstInternalFormat, GLImageFormat srcFormat,
        GLenum srcType, const void* srcValue )
{
    glClearNamedBufferData( m_id, static_cast< GLenum >( dstInternalFormat ),
        static_cast< GLenum >( srcFormat ), static_cast< GLenum >( srcType ),
        srcValue );
}

void GLBufferObject::clearRange( GLintptr offsetBytes, GLintptr sizeBytes )
{
    glClearNamedBufferSubData( m_id, GL_R8, offsetBytes, sizeBytes, GL_RED,
        GL_UNSIGNED_BYTE, NULL );
}

void GLBufferObject::clearRange( GLImageInternalFormat dstInternalFormat,
    GLImageFormat srcFormat, GLintptr dstOffsetBytes, GLintptr dstSizeBytes,
    GLenum srcType, const void* srcValue )
{
    glClearNamedBufferSubData( m_id, static_cast< GLenum >( dstInternalFormat ),
        dstOffsetBytes, dstSizeBytes,
        static_cast< GLenum >( srcFormat ), static_cast< GLenum >( srcType ),
        srcValue );
}

GLbitfield glStorageFlags( GLBufferObject::StorageFlags flags )
{
    return static_cast< GLbitfield >( flags );
}

GLBufferObject::StorageFlags operator | ( GLBufferObject::StorageFlags lhs,
                                           GLBufferObject::StorageFlags rhs )
{
    return static_cast< GLBufferObject::StorageFlags >(
        static_cast< GLbitfield >( lhs ) | static_cast< GLbitfield >( rhs )
    );
}

GLBufferObject::StorageFlags& operator |= (
    GLBufferObject::StorageFlags& lhs, GLBufferObject::StorageFlags rhs )
{
    lhs = static_cast< GLBufferObject::StorageFlags >(
        static_cast< GLbitfield >( lhs ) | static_cast< GLbitfield >( rhs )
    );
    return lhs;
}

GLbitfield glBufferMapRangeAccess( GLBufferObject::MapRangeAccess access )
{
    return static_cast< GLbitfield >( access );
}

GLBufferObject::MapRangeAccess operator | ( GLBufferObject::MapRangeAccess lhs,
                                           GLBufferObject::MapRangeAccess rhs )
{
    return static_cast< GLBufferObject::MapRangeAccess >(
        static_cast< GLbitfield >( lhs ) | static_cast< GLbitfield >( rhs )
    );
}

GLBufferObject::MapRangeAccess& operator |= (
    GLBufferObject::MapRangeAccess& lhs, GLBufferObject::MapRangeAccess rhs )
{
    lhs = static_cast< GLBufferObject::MapRangeAccess >(
        static_cast< GLbitfield >( lhs ) | static_cast< GLbitfield >( rhs )
    );
    return lhs;
}