#include "GLTexture3D.h"

#include <cassert>
#include <cstdio>
#include <vector>

#include "libcgt/core/math/Arithmetic.h"
#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Vector3f.h"

using libcgt::core::math::log2;
using libcgt::core::math::maximum;

// static
int GLTexture3D::calculateNumMipMapLevels( const Vector3i& size )
{
    return 1 + log2( maximum( size ) );
}

// static
Vector3i GLTexture3D::calculateMipMapSizeForLevel( const Vector3i& baseSize,
                                                  int level )
{
    if( level <= 0 )
    {
        return baseSize;
    }

    Vector3i size = baseSize;
    while( level > 0 )
    {
        size = maximum( Vector3i{ 1 }, size / 2 );
        --level;
    }
    return size;
}

GLTexture3D::GLTexture3D( const Vector3i& size,
                         GLImageInternalFormat internalFormat,
                         GLsizei nMipMapLevels ) :
    GLTexture( GLTexture::Target::TEXTURE_3D, internalFormat, nMipMapLevels ),
    m_size( size )
{
    assert( size.x > 0 );
    assert( size.y > 0 );
    assert( size.z > 0 );
    assert( size.x <= GLTexture::maxSize3D() );
    assert( size.y <= GLTexture::maxSize3D() );
    assert( size.z <= GLTexture::maxSize3D() );

    glTextureStorage3D( id(), 1, glInternalFormat(),
        size.x, size.y, size.z );
}

int GLTexture3D::numElements() const
{
    return m_size.x * m_size.y * m_size.z;
}

int GLTexture3D::width() const
{
    return m_size.x;
}

int GLTexture3D::height() const
{
    return m_size.y;
}

int GLTexture3D::depth() const
{
    return m_size.z;
}

Vector3i GLTexture3D::size() const
{
    return m_size;
}

bool GLTexture3D::set( Array3DReadView< uint8_t > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RED &&
       format != GLImageFormat::GREEN &&
       format != GLImageFormat::BLUE &&
       format != GLImageFormat::ALPHA )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::set( Array3DReadView< uint8x2 > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RG )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::set( Array3DReadView< uint8x3 > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RGB && format != GLImageFormat::BGR )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::set( Array3DReadView< uint8x4 > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RGBA && format != GLImageFormat::BGRA )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::set( Array3DReadView< float > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RED &&
       format != GLImageFormat::GREEN &&
       format != GLImageFormat::BLUE &&
       format != GLImageFormat::ALPHA )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::set( Array3DReadView< Vector2f > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RG )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::set( Array3DReadView< Vector3f > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RGB && format != GLImageFormat::BGR )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::set( Array3DReadView< Vector4f > data,
                      GLImageFormat format,
                      const Vector3i& dstOffset )
{
    if( format != GLImageFormat::RGBA && format != GLImageFormat::BGRA )
    {
        return false;
    }
    if( !data.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set3D( data.pointer(), data.size(), format, srcType, dstOffset );
}

bool GLTexture3D::checkSize( const Vector3i& srcSize,
                            const Vector3i& dstOffset ) const
{
    if( dstOffset.x + srcSize.x > m_size.x ||
       dstOffset.y + srcSize.y > m_size.y ||
       dstOffset.z + srcSize.z > m_size.z )
    {
        return false;
    }
    return true;
}

bool GLTexture3D::get3D( GLint srcLevel,
                        GLImageFormat dstFormat, GLPixelType dstType,
                        const Vector3i& dstSize, void* dstPtr )
{
    if( dstPtr == nullptr )
    {
        return false;
    }
    // TODO: rename checkSize to support srcLevel properly.
    bool succeeded = checkSize( dstSize, { 0, 0, 0 } );
    if( succeeded )
    {
        glGetTextureImage
        (
            id(), srcLevel,
            glImageFormat( dstFormat ), glPixelType( dstType ),
            dstSize.x * dstSize.y * dstSize.z,
            dstPtr
        );
        return true;
    }
    return succeeded;
}

bool GLTexture3D::set3D( const void* srcPtr, const Vector3i& srcSize,
                        GLImageFormat srcFormat, GLPixelType srcType,
                        const Vector3i& dstOffset )
{
    if( srcPtr == nullptr )
    {
        return false;
    }
    bool succeeded = checkSize( srcSize, dstOffset );
    if( succeeded )
    {
        glTextureSubImage3D
        (
            id(), 0,
            dstOffset.x, dstOffset.y, dstOffset.z,
            srcSize.x, srcSize.y, srcSize.z,
            glImageFormat( srcFormat ), glPixelType( srcType ),
            srcPtr
        );
    }
    return succeeded;
}
