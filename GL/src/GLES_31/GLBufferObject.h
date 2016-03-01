#pragma once

#include <cstdint>

#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

#include <common/BasicTypes.h>
#include <common/Array1DView.h>

#include "GLImageFormat.h"
#include "GLImageInternalFormat.h"

class GLBufferObject
{
public:

    // ***** Beware how ELEMENT_ARRAY_BUFFER works *****
    // ***** It requires a vertex array object to be bound ******
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

        TRANSFORM_FEEDBACK_BUFFER = GL_TRANSFORM_FEEDBACK_BUFFER,
        UNIFORM_BUFFER = GL_UNIFORM_BUFFER,

        // Compute
        ATOMIC_COUNTER_BUFFER = GL_ATOMIC_COUNTER_BUFFER,
        DISPATCH_INDIRECT_BUFFER = GL_DISPATCH_INDIRECT_BUFFER,
        SHADER_STORAGE_BUFFER = GL_SHADER_STORAGE_BUFFER,

        // Utility targets for copy()
        COPY_READ_BUFFER = GL_COPY_READ_BUFFER,
        COPY_WRITE_BUFFER = GL_COPY_WRITE_BUFFER
    };

#if 1
    // Mutable buffer storage.
    enum class Usage
    {
        STATIC_DRAW = GL_STATIC_DRAW,
        STATIC_READ = GL_STATIC_READ,
        STATIC_COPY = GL_STATIC_COPY,

        DYNAMIC_DRAW = GL_DYNAMIC_DRAW,
        DYNAMIC_READ = GL_DYNAMIC_READ,
        DYNAMIC_COPY = GL_DYNAMIC_COPY,

        STREAM_DRAW = GL_STREAM_DRAW,
        STREAM_READ = GL_STREAM_READ,
        STREAM_COPY = GL_STREAM_COPY
    };
#endif

#if 0
// For EXT_buffer_storage
    enum class StorageFlags : GLbitfield
    {
        MAP_READ_BIT = GL_MAP_READ_BIT,
        MAP_WRITE_BIT = GL_MAP_WRITE_BIT,
        MAP_PERSISTENT_BIT = GL_MAP_PERSISTENT_BIT_EXT,
        MAP_COHERENT_BIT = GL_MAP_COHERENT_BIT_EXT,
        DYNAMIC_STORAGE_BIT = GL_DYNAMIC_STORAGE_BIT_EXT,
        CLIENT_STORAGE_BIT = GL_CLIENT_STORAGE_BIT_EXT
    };
#endif

    enum class MapRangeAccess : GLbitfield
    {
        READ_BIT = GL_MAP_READ_BIT,
        WRITE_BIT = GL_MAP_WRITE_BIT,
        INVALIDATE_RANGE_BIT = GL_MAP_INVALIDATE_RANGE_BIT,
        INVALIDATE_BUFFER_BIT = GL_MAP_INVALIDATE_BUFFER_BIT,
#if 0
// For EXT_buffer_storage
        FLUSH_EXPLICIT_BIT = GL_MAP_FLUSH_EXPLICIT_BIT_EXT,
        UNSYNCHRONIZED_BIT = GL_MAP_UNSYNCHRONIZED_BIT_EXT,
        PERSISTENT_BIT = GL_MAP_PERSISTENT_BIT_EXT,
        COHERENT_BIT = GL_MAP_COHERENT_BIT_EXT
#endif
    };

    // TODO: copy() functionality

    // Construct a buffer object of the specified size.
    // Target is the initial target to which the buffer will be bound and
    // unbound for construction.
    // Usage is the intended usage for the buffer.
    GLBufferObject( GLsizeiptr nBytes, Target target, Usage usage );

    // Construct a buffer object with the specified initial data and size.
    // data must be packed().
    // Target is the initial target to which the buffer will be bound and
    // unbound for construction.
    // Usage is the intended usage for the buffer.
    GLBufferObject( Array1DView< const uint8_t > data, Target target,
                   Usage usage );

    virtual ~GLBufferObject();

    GLuint id() const;

    // Total bytes allocated.
    size_t numBytes() const;

    // The usage specified upon creation.
    Usage usage() const;

    // Bind this buffer object to the given target.
    // A buffer may be bound to multiple targets at once.
    //
    // To map a buffer, first bind it to a target. Then call mapRange() on a
    // that target. Then unbind.
    // TODO: unbinding is not necessary once EXT_buffer_storage is supported.
    void bind( Target target );

    // Unbind the buffer that is currently bound to the given target.
    static void unbind( Target target );

    // *If* this buffer is bound to Target target, then:
    //
    // Maps a range of this buffer as a pointer into client memory.
    // access is a bitfield, which is an OR combination of elements of
    // MapRangeAccess.
    //   It must have one of GL_MAP_READ_BIT, or GL_MAP_WRITE_BIT.
    //   It can also optionally have GL_MAP_INVALIDATE_RANGE_BIT,
    //     GL_MAP_INVALIDATE_BUFFER_BIT, GL_MAP_FLUSH_EXPLICIT_BIT,
    //     and/or GL_MAP_UNSYNCHRONIZED_BIT.
    // TODO: make a Range1l class for 64-bit ranges (and use it here).
    Array1DView< uint8_t > mapRange( Target target,
                                    GLintptr offsetBytes, GLsizeiptr sizeBytes,
                                    MapRangeAccess access );

    // *If* this buffer is bound to Target target, then:
    //
    // Maps a range of this buffer as a pointer into client memory.
    // access is a bitfield, which is an OR combination of elements of
    // MapRangeAccess.
    //   It must have one of GL_MAP_READ_BIT, or GL_MAP_WRITE_BIT.
    //   It can also optionally have GL_MAP_INVALIDATE_RANGE_BIT,
    //     GL_MAP_INVALIDATE_BUFFER_BIT, GL_MAP_FLUSH_EXPLICIT_BIT,
    //     and/or GL_MAP_UNSYNCHRONIZED_BIT.
    // TODO: make a Range1l class for 64-bit ranges (and use it here).
    template< typename T >
    Array1DView< T > mapRangeAs( Target target,
                                GLintptr offsetBytes, GLsizeiptr sizeBytes,
                                MapRangeAccess access );

    // If the buffer was mapped with GL_MAP_FLUSH_EXPLICIT_BIT, flushRange()
    // tells OpenGL that this range should now be visible to calls after this.
    // TODO: consider redesigning this interface to return mapped range objects.
    void flushRange( Target target, GLintptr offsetBytes, GLintptr sizeBytes );

    // Unmap the previously mapped buffer, which is currently bound to target.
    static void unmap( Target target );

    // Copies the data from src into this buffer.
    // Returns false if:
    // The source buffer is not packed(), or
    // if dstOffset + src.length() > numBytes().
    bool set( Target target, Array1DView< const uint8_t > src,
             GLintptr dstOffset );

    // TODO: persistent mapping:
    // glMemoryBarrier( GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT ),

private:

    GLuint m_id;
    GLsizeiptr m_nBytes;
    Usage m_usage;
};

GLenum glBufferTarget( GLBufferObject::Target target );
GLenum glBufferUsage( GLBufferObject::Usage usage );
GLbitfield glBufferMapRangeAccess( GLBufferObject::MapRangeAccess access );

GLBufferObject::MapRangeAccess operator | ( GLBufferObject::MapRangeAccess lhs,
    GLBufferObject::MapRangeAccess rhs );
GLBufferObject::MapRangeAccess& operator |= (
    GLBufferObject::MapRangeAccess& lhs, GLBufferObject::MapRangeAccess rhs );

template< typename T >
Array1DView< T > GLBufferObject::mapRangeAs( GLBufferObject::Target target,
    GLintptr offsetBytes, GLsizeiptr sizeBytes,
    GLBufferObject::MapRangeAccess access )
{
    return Array1DView< T >(
        glMapBufferRange( glBufferTarget( target ),
             offsetBytes, sizeBytes, glBufferMapRangeAccess( access ) ),
        static_cast< size_t >( sizeBytes / sizeof( T ) )
    );
}