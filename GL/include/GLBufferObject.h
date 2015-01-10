#pragma once

#include <cstdint>

#include <GL/glew.h>
#include <vector>

#include <common/BasicTypes.h>
#include <common/Array1DView.h>

class GLBufferObject
{
public:

	enum class Target
	{
		NONE = 0,

		// Vertex Buffer Objects
		ARRAY_BUFFER = GL_ARRAY_BUFFER,
		ELEMENT_ARRAY_BUFFER = GL_ELEMENT_ARRAY_BUFFER,
		DRAW_INDIRECT_BUFFER = GL_DRAW_INDIRECT_BUFFER,

		// Pixel Buffer Objects
		PIXEL_PACK_BUFFER = GL_PIXEL_PACK_BUFFER,
		PIXEL_UNPACK_BUFFER = GL_PIXEL_UNPACK_BUFFER,
		
		QUERY_BUFFER = GL_QUERY_BUFFER,
		TRANSFORM_FEEDBACK_BUFFER = GL_TRANSFORM_FEEDBACK_BUFFER,
		UNIFORM_BUFFER = GL_UNIFORM_BUFFER,
		TEXTURE_BUFFER = GL_TEXTURE_BUFFER,

		// Compute
		ATOMIC_COUNTER_BUFFER = GL_ATOMIC_COUNTER_BUFFER,
		DISPATCH_INDIRECT_BUFFER = GL_DISPATCH_INDIRECT_BUFFER,
		SHADER_STORAGE_BUFFER = GL_SHADER_STORAGE_BUFFER
	};

    enum class Access
	{
		READ_ONLY = GL_READ_ONLY,
		WRITE_ONLY = GL_WRITE_ONLY,
		READ_WRITE = GL_READ_WRITE
	};

	// Direct copy between two buffer objects.
    // Returns false if either range is out of bounds.
	static bool copy( GLBufferObject* pSource, GLBufferObject* pDestination,
		GLintptr sourceOffsetBytes, GLintptr destinationOffsetBytes,
		GLsizeiptr nBytes );

	// Construct a buffer object with capacity nBytes, access permissions in flags,
	// and optional initial data. If data is nullptr (default), then it creates an empty
	// buffer with undefined data.
	//
	// flags is an OR mask of:
	// GL_MAP_READ_BIT​: user can read using map() (glMapBuffer)
	// GL_MAP_WRITE_BIT​: user can write using map() (glMapBuffer)
	// GL_DYNAMIC_STORAGE_BIT​: user can modify using set() (glBufferSubData)
	// GL_PERSISTENT_BIT​: can be used while mapped.
	// GL_COHERENT_BIT​: allows read and write while mapped without barriers.
	//   *Requires GL_PERSISTENT_BIT.*
	// GL_CLIENT_STORAGE_BIT​: hint that the memory should live in client memory.
	//
	// Userful combinations:
	// Pure OpenGL: set flags to 0.
	// Static vertex data: set flags to 0 with initial data.
	// Read-back only: use GL_MAP_READ_BIT and map(), glGetBufferSubData() isn't very efficient.
	// Updatable: use GL_MAP_WRITE_BIT and map(), glBufferSubData() isn't very efficient.
	GLBufferObject( size_t nBytes,
		GLbitfield flags, const void* data = nullptr );

	// destroy a GLBufferObject
	virtual ~GLBufferObject();

	GLuint id() const;

	// Total bytes allocated.
	size_t numBytes() const;

	// Map the entire buffer into client memory.
    Array1DView< uint8_t > map( Access access );

	// Map the entire buffer into client memory, with length
	//   numBytes() / sizeof( T ).
	template< typename T >
    Array1DView< T > mapAs( Access access );

	// Map part of this buffer as a pointer into client memory.
    Array1DView< uint8_t > mapRange( GLintptr offset, GLsizeiptr length, Access access );

	// Map part of this buffer as a pointer into client memory.	
	//   byteOffset is obviously *in bytes*,
	//   nElements is the number of elements of type T
    // TODO: make a Range1i class, just like Rect2i. For 64-bit, make Range1l
	template< typename T >
    Array1DView< T > mapRangeAs( GLintptr byteOffset, size_t nElements, Access access );

	// Unmap this buffer from local memory.
	void unmap();
	
	// Copies the data in a subset of this buffer to dst.
	// Returns false if srcOffset + dst.length() > numBytes().
	bool get( GLintptr srcOffset, Array1DView< uint8_t > dst );

	// Copies the data from src into this buffer.
	// Returns false if:
	// The source buffer is not packed(), or
	// if dstOffset + src.length() > numBytes().
	bool set( Array1DView< const uint8_t > src, GLintptr dstOffset );

	// TODO: glClearNamedBufferDataEXT / glClearNamedBufferSubDataEXT
	// TODO: persistent mapping:
	//  glMemoryBarrier( GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT ),
	//  glFlushMappedBufferRange,
	//  glInvalidateBufferData / glInvalidateBufferSubData

private:

	GLuint m_id;
	size_t m_nBytes;
};

template< typename T >
Array1DView< T > GLBufferObject::mapAs( GLBufferObject::Access access )
{
    return Array1DView< T >(
        glMapNamedBufferEXT( m_id, static_cast< GLbitfield >( access ) ),
        m_nBytes / sizeof( T ) );
}

template< typename T >
Array1DView< T > GLBufferObject::mapRangeAs( GLintptr byteOffset, size_t nElements,
    GLBufferObject::Access access )
{
	GLsizeiptr nBytes = nElements * sizeof( T );
    return Array1DView< T >(
        glMapNamedBufferRangeEXT( m_id, byteOffset, nBytes, static_cast< GLbitfield >( access ) ),
        nElements );
}
