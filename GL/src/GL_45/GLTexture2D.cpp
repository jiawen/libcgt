#include "GLTexture2D.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <GL/glew.h>

#include <math/Arithmetic.h>
#include <math/MathUtils.h>
#include <vecmath/Vector3f.h>

#include "GLSamplerObject.h"

// static
int GLTexture2D::calculateNumMipMapLevels( const Vector2i& size )
{
    return 1 + Arithmetic::log2( MathUtils::maximum( size ) );
}

// static
Vector2i GLTexture2D::calculateMipMapSizeForLevel( const Vector2i& baseSize,
    int level )
{
    if( level <= 0 )
    {
        return baseSize;
    }

    Vector2i size = baseSize;
    while( level > 0 )
    {
        size = MathUtils::maximum( Vector2i{ 1, 1 }, size / 2 );
        --level;
    }
    return size;
}

GLTexture2D::GLTexture2D( const Vector2i& size,
    GLImageInternalFormat internalFormat, GLsizei nMipMapLevels ) :
    GLTexture( GLTexture::Target::TEXTURE_2D, internalFormat, nMipMapLevels ),
    m_size( size )
{
    assert( size.x > 0 );
    assert( size.y > 0 );
    assert( size.x <= GLTexture::maxSize2D() );
    assert( size.y <= GLTexture::maxSize2D() );

    glTextureStorage2D( id(), numMipMapLevels(), glInternalFormat(),
                       size.x, size.y );
}

int GLTexture2D::width() const
{
    return m_size.x;
}

int GLTexture2D::height() const
{
    return m_size.y;
}

Vector2i GLTexture2D::size() const
{
    return m_size;
}

bool GLTexture2D::get( Array2DView< uint8_t > output, GLImageFormat dstFormat )
{
    if( !output.packed() )
    {
        return false;
    }
    const GLPixelType dstType = GLPixelType::UNSIGNED_BYTE;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );
}

bool GLTexture2D::get( Array2DView< uint8x2 > output )
{
    if( !output.packed() )
    {
        return false;
    }
    const GLImageFormat dstFormat = GLImageFormat::RG;
    const GLPixelType dstType = GLPixelType::UNSIGNED_BYTE;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );
}

bool GLTexture2D::get( Array2DView< uint8x3 > output,
                      GLImageFormat dstFormat )
{
    if( !output.packed() )
    {
        return false;
    }
    if( dstFormat != GLImageFormat::RGB &&
       dstFormat != GLImageFormat::BGR )
    {
        return false;
    }
    const GLPixelType dstType = GLPixelType::UNSIGNED_BYTE;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );

}

bool GLTexture2D::get( Array2DView< uint8x4 > output,
                      GLImageFormat dstFormat )
{
    if( !output.packed() )
    {
        return false;
    }
    if( dstFormat != GLImageFormat::RGBA &&
       dstFormat != GLImageFormat::BGRA )
    {
        return false;
    }
    const GLPixelType dstType = GLPixelType::UNSIGNED_BYTE;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );
}

bool GLTexture2D::get( Array2DView< float > output,
                      GLImageFormat dstFormat )
{
    if( !output.packed() )
    {
        return false;
    }
    if( dstFormat != GLImageFormat::RED &&
       dstFormat != GLImageFormat::GREEN &&
       dstFormat != GLImageFormat::BLUE &&
       dstFormat != GLImageFormat::ALPHA )
    {
        return false;
    }
    const GLPixelType dstType = GLPixelType::FLOAT;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );

}

bool GLTexture2D::get( Array2DView< Vector2f > output )
{
    if( !output.packed() )
    {
        return false;
    }
    const GLImageFormat dstFormat = GLImageFormat::RG;
    const GLPixelType dstType = GLPixelType::FLOAT;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );

}

bool GLTexture2D::get( Array2DView< Vector3f > output,
                      GLImageFormat dstFormat )
{
    if( !output.packed() )
    {
        return false;
    }
    if( dstFormat != GLImageFormat::RGB && dstFormat != GLImageFormat::BGR )
    {
        return false;
    }
    const GLPixelType dstType = GLPixelType::FLOAT;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );
}

bool GLTexture2D::get( Array2DView< Vector4f > output,
                      GLImageFormat dstFormat )
{
    if( !output.packed() )
    {
        return false;
    }
    if( dstFormat != GLImageFormat::RGBA &&
       dstFormat != GLImageFormat::BGRA )
    {
        return false;
    }
    const GLPixelType dstType = GLPixelType::FLOAT;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );

}

bool GLTexture2D::set( Array2DView< const uint8_t > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RED &&
       srcFormat != GLImageFormat::GREEN &&
       srcFormat != GLImageFormat::BLUE &&
       srcFormat != GLImageFormat::ALPHA )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture2D::set( Array2DView< const uint8x2 > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RG )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture2D::set( Array2DView< const uint8x3 > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RGB &&
       srcFormat != GLImageFormat::BGR )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture2D::set( Array2DView< const uint8x4 > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RGBA && srcFormat != GLImageFormat::BGRA )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture2D::set( Array2DView< const float > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RED &&
       srcFormat != GLImageFormat::GREEN &&
       srcFormat != GLImageFormat::BLUE &&
       srcFormat != GLImageFormat::ALPHA )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture2D::set( Array2DView< const Vector2f > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RG )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture2D::set( Array2DView< const Vector3f > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RGB && srcFormat != GLImageFormat::BGR )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture2D::set( Array2DView< const Vector4f > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    if( srcFormat != GLImageFormat::RGBA && srcFormat != GLImageFormat::BGRA )
    {
        return false;
    }
    if( !srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set2D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

void GLTexture2D::drawNV( GLSamplerObject* pSampler,
                         float z,
                         const Rect2f& texCoords )
{
    drawNV( Rect2f( size() ), pSampler, z, texCoords );
}

void GLTexture2D::drawNV( const Rect2f& windowCoords,
                         GLSamplerObject* pSampler,
                         float z,
                         const Rect2f& texCoords )
{
    GLuint samplerId = pSampler == nullptr ? 0 : pSampler->id();

    Vector2f p0 = windowCoords.origin();
    Vector2f p1 = windowCoords.limit();

    Vector2f t0 = texCoords.origin();
    Vector2f t1 = texCoords.limit();

    glDrawTextureNV
    (
        id(), samplerId,
        p0.x, p0.y,
        p1.x, p1.y,
        z,
        t0.x, t0.y,
        t1.x, t1.y
    );
}

bool GLTexture2D::checkSize( const Vector2i& srcSize,
                            const Vector2i& dstOffset ) const
{
    if( dstOffset.x + srcSize.x > m_size.x ||
       dstOffset.y + srcSize.y > m_size.y )
    {
        return false;
    }
    return true;
}

bool GLTexture2D::get2D( GLint srcLevel,
                        GLImageFormat dstFormat, GLPixelType dstType,
                        const Vector2i& dstSize, void* dstPtr )
{
    if( dstPtr == nullptr )
    {
        return false;
    }
    // TODO: rename checkSize to support srcLevel properly.
    bool succeeded = checkSize( dstSize, { 0, 0 } );
    if( succeeded )
    {
        glGetTextureImage
        (
            id(), srcLevel,
            glImageFormat( dstFormat ), glPixelType( dstType ),
            dstSize.x * dstSize.y,
            dstPtr
        );
        return true;
    }
    return succeeded;
}

bool GLTexture2D::set2D( const void* srcPtr, const Vector2i& srcSize,
                        GLImageFormat srcFormat, GLPixelType srcType,
                        const Vector2i& dstOffset )
{
    if( srcPtr == nullptr )
    {
        return false;
    }
    bool succeeded = checkSize( srcSize, dstOffset );
    if( succeeded )
    {
        glTextureSubImage2D
        (
            id(), 0, // TODO: level
            dstOffset.x, dstOffset.y, srcSize.x, srcSize.y,
            glImageFormat( srcFormat ), glPixelType( srcType ),
            srcPtr
        );
    }
    return succeeded;
}
