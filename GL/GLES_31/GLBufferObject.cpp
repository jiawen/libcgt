#include <cassert>
#include <common/ArrayUtils.h>

#include "GLBufferObject.h"

GLBufferObject::GLBufferObject( GLsizeiptr nBytes,
    GLBufferObject::Target target,
    GLBufferObject::Usage usage ) :
    m_id( 0 ),
    m_nBytes( nBytes ),
    m_usage( usage )
{
    glGenBuffers( 1, &m_id );

    bind( target );
    glBufferData( glBufferTarget( target ), nBytes, nullptr,
                 glBufferUsage( usage ) );
    unbind( target );
}

GLBufferObject::GLBufferObject( Array1DView< const uint8_t > data,
    GLBufferObject::Target target,
    GLBufferObject::Usage usage ) :
    m_id( 0 ),
    m_nBytes( data.size() ),
    m_usage( usage )
{
    glGenBuffers( 1, &m_id );

    if( data.notNull() && data.packed() )
    {
        glGenBuffers( 1, &m_id );

        bind( target );
        glBufferData( glBufferTarget( target ), m_nBytes, nullptr,
                     glBufferUsage( usage ) );
        unbind( target );
    }
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

GLBufferObject::Usage GLBufferObject::usage() const
{
    return m_usage;
}

void GLBufferObject::bind( GLBufferObject::Target target )
{
    glBindBuffer( glBufferTarget( target ), id() );
}

// static
void GLBufferObject::unbind( GLBufferObject::Target target )
{
    glBindBuffer( glBufferTarget( target ), 0 );
}

Array1DView< uint8_t > GLBufferObject::mapRange( GLBufferObject::Target target,
    GLintptr offsetBytes, GLsizeiptr sizeBytes,
    GLBufferObject::MapRangeAccess access )
{
    return Array1DView< uint8_t >(
        glMapBufferRange( glBufferTarget( target ),
            offsetBytes, sizeBytes, glBufferMapRangeAccess( access ) ),
        sizeBytes );
}

void GLBufferObject::flushRange( GLBufferObject::Target target,
    GLintptr offsetBytes, GLintptr sizeBytes )
{
    glFlushMappedBufferRange( glBufferTarget( target ), offsetBytes,
        sizeBytes );
}

void GLBufferObject::unmap( GLBufferObject::Target target )
{
    glUnmapBuffer( glBufferTarget( target ) );
}

bool GLBufferObject::set( GLBufferObject::Target target,
    Array1DView< const uint8_t > src, GLintptr dstOffset )
{
    if( !( src.packed() ) )
    {
        return false;
    }
    if( dstOffset + src.size() > m_nBytes )
    {
        return false;
    }

    glBufferSubData( glBufferTarget( target ), dstOffset, src.size(),
                    src.pointer() );
    return true;
}

GLenum glBufferTarget( GLBufferObject::Target target )
{
    return static_cast< GLenum >( target );
}

GLenum glBufferUsage( GLBufferObject::Usage usage )
{
    return static_cast< GLenum >( usage );
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