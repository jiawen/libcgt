#pragma once

#include <cstdint>

#include <GL/glew.h>
#include <vector>

#include <common/BasicTypes.h>
#include <common/Array1DView.h>

#include "GLImageFormat.h"
#include "GLImageInternalFormat.h"

// A "mutable data store" in the OpenGL sense is explicitly not supported.
// I.e., no call to glNamedBufferData().
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

    // Construct a buffer object with access permissions in flags.
    // In the first form, creates a buffer with capacity nBytes and undefined
    // contents.
    // In the second form, creates a buffer exactly matching the input data.
    //
    // flags is an OR mask of:
    // GL_MAP_READ_BIT: user can read using map() (glMapBuffer)
    // GL_MAP_WRITE_BIT: user can write using map() (glMapBuffer)
    // GL_DYNAMIC_STORAGE_BIT: user can modify using set() (glBufferSubData)
    // GL_PERSISTENT_BIT: can be used while mapped.
    // GL_COHERENT_BIT: allows read and write while mapped without barriers.
    //   *Requires GL_PERSISTENT_BIT.*
    // GL_CLIENT_STORAGE_BIT: hint that the memory should live in client memory.
    //
    // Userful combinations:
    // Pure OpenGL: set flags to 0.
    // Static vertex data: set flags to 0 with initial data.
    // Read-back only: use GL_MAP_READ_BIT and map(), glGetBufferSubData() isn't very efficient.
    // Updatable: use GL_MAP_WRITE_BIT and map(), glBufferSubData() isn't very efficient.
    GLBufferObject( GLbitfield flags, GLsizeiptr nBytes );
    // TODO(size): Array1DView.size needs to be a size_t.
    GLBufferObject( GLbitfield flags, Array1DView< uint8_t > data );

    // destroy a GLBufferObject
    virtual ~GLBufferObject();

    GLuint id() const;

    // Total bytes allocated.
    size_t numBytes() const;

    // Map the entire buffer into client memory.
    Array1DView< uint8_t > map( Access access );

    // Map the entire buffer into client memory, with length
    // numBytes() / sizeof( T ).
    template< typename T >
    Array1DView< T > mapAs( Access access );

    // Map a range of this buffer as a pointer into client memory.
    // access is a bitfield:
    //   It must have one of GL_MAP_READ_BIT, or GL_MAP_WRITE_BIT.
    //   It can be GL_MAP_PERSISTENT_BIT and/or GL_MAP_COHERENT_BIT.
    //   It can also optionally have GL_MAP_INVALIDATE_RANGE_BIT,
    //     GL_MAP_INVALIDATE_BUFFER_BIT, GL_MAP_FLUSH_EXPLICIT_BIT,
    //     and/or GL_MAP_UNSYNCHRONIZED_BIT.
    Array1DView< uint8_t > mapRange( GLintptr offsetBytes, GLsizeiptr sizeBytes, GLbitfield access );

    // Map part of this buffer as a pointer into client memory.
    //   byteOffset is the offset from the beginning of the buffer,
    //     in bytes.
    //   sizeBytes is the number of bytes in the range.
    // access is a bitfield:
    //   It must have one of GL_MAP_READ_BIT, or GL_MAP_WRITE_BIT.
    //   It can be GL_MAP_PERSISTENT_BIT and/or GL_MAP_COHERENT_BIT.
    //   It can also optionally have GL_MAP_INVALIDATE_RANGE_BIT,
    //     GL_MAP_INVALIDATE_BUFFER_BIT, GL_MAP_FLUSH_EXPLICIT_BIT,
    //     and/or GL_MAP_UNSYNCHRONIZED_BIT.
    // TODO: make a Range2l class for 64-bit ranges (and use it here).
    template< typename T >
    Array1DView< T > mapRangeAs( GLintptr offsetBytes, GLsizeiptr sizeBytes, GLbitfield access );

    // If the buffer was mapped with GL_MAP_FLUSH_EXPLICIT_BIT, flushRange()
    // tells OpenGL that this range should now be visible to calls after this.
    void flushRange( GLintptr offsetBytes, GLintptr sizeBytes );

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

    // Fills the entire buffer with 0.
    void clear();

    // Clears the entire buffer to a single value.
    // The buffer's contents are written in the internal format specified.
    // The input is considered srcFormat with type srcType.
    // srcValue is a single element of this (type, format).
    // Note that not all internal formats and source input formats are allowed.
    // Consult:
    // https://www.opengl.org/registry/specs/ARB/clear_buffer_object.txt for up
    // to date information.
    void clear( GLImageInternalFormat dstInternalFormat, GLImageFormat srcFormat,
        GLenum srcType, const void* srcValue );

    void clearRange( GLintptr offset, GLintptr size );
    void clearRange( GLImageInternalFormat dstInternalFormat, GLImageFormat srcFormat,
        GLintptr dstOffsetBytes, GLintptr dstSizeBytes, GLenum srcType, const void* srcValue );

    // Give a hint to the implementation to invalidate the old contents of the
    // buffer. Pending reads issued before this call will still get the old
    // values, but later calls are not guaranteed. You are expected to set new
    // data (via set or map) before reading again.
    void invalidate();

    // Same as invalidate() but on a range.
    void invalidateRange( GLintptr offsetBytes, GLintptr sizeBytes );

    // TODO: persistent mapping:
    // glMemoryBarrier( GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT ),

    // TODO: nice to have:
    // glGetNamedBufferParameteriv(), glGetNamedBufferParameteri64v(): to retrieve
    // parameters about the buffer object.
    // glGetNamedBufferPointerv(): not very useful. If it's mapped and you lost the pointer,
    // call this to get it again.

private:

    GLuint m_id;
    GLsizeiptr m_nBytes;
};

template< typename T >
Array1DView< T > GLBufferObject::mapAs( GLBufferObject::Access access )
{
    return Array1DView< T >(
        glMapNamedBuffer( m_id, static_cast< GLenum >( access ) ),
        static_cast< int >( m_nBytes / sizeof( T ) ) );
}

template< typename T >
Array1DView< T > GLBufferObject::mapRangeAs( GLintptr offsetBytes, GLsizeiptr sizeBytes,
    GLbitfield access )
{
    return Array1DView< T >(
        glMapNamedBufferRange( m_id, offsetBytes, sizeBytes, access ),
        static_cast< int >( sizeBytes / sizeof( T ) ) );
}
