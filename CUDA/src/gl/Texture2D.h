#pragma once

#include <GLTexture2D.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace libcgt { namespace cuda { namespace gl {

// TODO: consider subclassing GLTexture2D instead.
// But first, consider refactoring GLTexture2D so that it uses the CRTP.
class Texture2D
{
public:

    enum class MapFlags
    {
        NONE = cudaGraphicsMapFlagsNone, // Denotes read-write.
        READ_ONLY = cudaGraphicsMapFlagsReadOnly,
        WRITE_DISCARD = cudaGraphicsMapFlagsWriteDiscard
    };

    class MappedResource
    {
    public:

        ~MappedResource();
        cudaArray* array() const;

    private:
        friend class libcgt::cuda::gl::Texture2D;
        cudaGraphicsResource* m_resource;
        cudaArray* m_array = nullptr;
    };

    Texture2D( GLTexture2D&& texture, MapFlags mapFlags = MapFlags::NONE );
    Texture2D( Texture2D&& move );
    Texture2D& operator = ( Texture2D&& move );
    ~Texture2D();

    const GLTexture2D& texture() const;
    GLTexture2D& texture();

    MappedResource map( unsigned int arrayIndex = 0, unsigned int mipLevel = 0,
        cudaStream_t stream = cudaStreamDefault );

private:

    GLTexture2D m_texture;
    cudaGraphicsResource* m_resource = nullptr;
};

} } } // gl, cuda, libcgt
