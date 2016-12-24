#include "Texture2D.h"

#include <cassert>

#include "libcgt/cuda/DeviceArray1D.h"
#include "libcgt/cuda/DeviceOpaqueArray2D.h"
#include "libcgt/cuda/DeviceArray2D.h"
#include "libcgt/cuda/DeviceArray3D.h"

namespace libcgt { namespace cuda { namespace gl {

Texture2D::Texture2D( GLTexture2D&& texture, MapFlags mapFlags ) :
    m_texture( std::move( texture ) )
{
    cudaError_t err = cudaGraphicsGLRegisterImage( &m_resource,
        m_texture.id(), GL_TEXTURE_2D,
        static_cast< unsigned int >( mapFlags ) );
    assert( err == cudaSuccess );
}

Texture2D::Texture2D( Texture2D&& move )
{
    m_texture = std::move( move.m_texture );
    m_resource = move.m_resource;
    move.m_resource = nullptr;
}

Texture2D& Texture2D::operator = ( Texture2D&& move )
{
    if( this != &move )
    {
        m_texture = std::move( move.m_texture );
        m_resource = move.m_resource;
        move.m_resource = nullptr;
    }
    return *this;
}

Texture2D::~Texture2D()
{
    if( m_resource != nullptr )
    {
        cudaError_t err = cudaGraphicsUnregisterResource( m_resource );
        assert( err == cudaSuccess );
        m_resource = nullptr;
    }
}

const GLTexture2D& Texture2D::texture() const
{
    return m_texture;
}

GLTexture2D& Texture2D::texture()
{
    return m_texture;
}

Texture2D::MappedResource::~MappedResource()
{
    cudaGraphicsUnmapResources( 1, &m_resource );
}

cudaArray* Texture2D::MappedResource::array() const
{
    return m_array;
}

Texture2D::MappedResource Texture2D::map( unsigned int arrayIndex,
    unsigned int mipLevel, cudaStream_t stream )
{
    cudaError_t err;
    MappedResource mr;
    mr.m_resource = m_resource;

    err = cudaGraphicsMapResources( 1, &m_resource, stream );
    assert( err == cudaSuccess );

    err = cudaGraphicsSubResourceGetMappedArray( &mr.m_array, mr.m_resource,
        arrayIndex, mipLevel );
    assert( err == cudaSuccess );

    // TODO: implement mip mapped array levels as needed.
    // cudaGraphicsResourceGetMappedMipmappedArray().

    return mr;
}

} } } // gl, cuda, libcgt
