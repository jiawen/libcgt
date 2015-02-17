#include <cassert>
#include <common/ArrayUtils.h>

#include "GLBufferObject.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
bool GLBufferObject::copy( GLBufferObject* pSource, GLBufferObject* pDestination,
						  GLintptr sourceOffsetBytes, GLintptr destinationOffsetBytes,
						  GLsizeiptr nBytes )
{
	if( sourceOffsetBytes >= pSource->numBytes() ||
		destinationOffsetBytes >= pDestination->numBytes() ||
		sourceOffsetBytes + nBytes >= pSource->numBytes() ||
		destinationOffsetBytes + nBytes >= pDestination->numBytes() )
	{
		return false;
	}

	glNamedCopyBufferSubDataEXT( pSource->m_id, pDestination->m_id,
		sourceOffsetBytes, destinationOffsetBytes,
		nBytes );
	return true;
}

GLBufferObject::GLBufferObject( size_t nBytes,
							   GLbitfield flags,
							   const void* data ) :

	m_id( 0 ),
	m_nBytes( nBytes )

{
	// printf( "GLBufferObject: allocating %f megabytes\n", nElements * bytesPerElement / 1048576.f );

	// TODO: OpenGL 4.4: move to glBufferStorage and different flags
	// http://www.opengl.org/wiki/GLAPI/glBufferStorage

	glGenBuffers( 1, &m_id );
	glNamedBufferStorageEXT( m_id, nBytes, data, flags );
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

Array1DView< uint8_t > GLBufferObject::map( Access access )
{
    return Array1DView< uint8_t >(
        glMapNamedBufferEXT( m_id, static_cast< GLenum >( access ) ),
        m_nBytes );
}

Array1DView< uint8_t > GLBufferObject::mapRange( GLintptr offset, GLsizeiptr length, GLbitfield access )
{
    return Array1DView< uint8_t >(
        glMapNamedBufferRangeEXT( m_id, offset, length, access ),
        length );
}

void GLBufferObject::unmap()
{
	glUnmapNamedBufferEXT( m_id );
}

bool GLBufferObject::get( GLintptr srcOffset, Array1DView< uint8_t > dst )
{
	if( srcOffset + dst.size() > m_nBytes )
	{
		return false;
	}

	glGetNamedBufferSubDataEXT( m_id, srcOffset, dst.size(), dst.pointer() );
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
	
	glNamedBufferSubDataEXT( m_id, dstOffset, src.size(), src.pointer() );
	return true;
}
