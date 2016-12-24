#include "GLTexture2D.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <GL/glew.h>

#include <math/Arithmetic.h>
#include <math/MathUtils.h>
#include <vecmath/Vector3f.h>

#include "libcgt/GL/GLSamplerObject.h"

using libcgt::core::math::log2;
using libcgt::core::math::maximum;

// static
int GLTexture2D::calculateNumMipMapLevels( const Vector2i& size )
{
    return 1 + log2( maximum( size ) );
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
        size = maximum( Vector2i{ 1 }, size / 2 );
        --level;
    }
    return size;
}

GLTexture2D::GLTexture2D() :
    GLTexture( GLTexture::Target::TEXTURE_2D, GLImageInternalFormat::NONE, 0 )
{

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

GLTexture2D::GLTexture2D( GLTexture2D&& move ) :
    GLTexture( std::move( move ) )
{
    m_size = move.m_size;
    move.m_size = { 0, 0 };
}

GLTexture2D& GLTexture2D::operator = ( GLTexture2D&& move )
{
    GLTexture::operator = ( std::move( move ) );
    if( this != &move )
    {
        m_size = move.m_size;
        move.m_size = { 0, 0 };
    }
    return *this;
}

// virtual
GLTexture2D::~GLTexture2D()
{
    m_size = { 0, 0 };
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

bool GLTexture2D::get( Array2DWriteView< uint8_t > output,
    GLImageFormat dstFormat )
{
    if( !output.packed() )
    {
        return false;
    }
    const GLPixelType dstType = GLPixelType::UNSIGNED_BYTE;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );
}

bool GLTexture2D::get( Array2DWriteView< uint8x2 > output )
{
    if( !output.packed() )
    {
        return false;
    }
    const GLImageFormat dstFormat = GLImageFormat::RG;
    const GLPixelType dstType = GLPixelType::UNSIGNED_BYTE;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );
}

bool GLTexture2D::get( Array2DWriteView< uint8x3 > output,
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

bool GLTexture2D::get( Array2DWriteView< uint8x4 > output,
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

bool GLTexture2D::get( Array2DWriteView< float > output,
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

bool GLTexture2D::get( Array2DWriteView< Vector2f > output )
{
    if( !output.packed() )
    {
        return false;
    }
    const GLImageFormat dstFormat = GLImageFormat::RG;
    const GLPixelType dstType = GLPixelType::FLOAT;
    return get2D( 0, dstFormat, dstType, output.size(), output.pointer() );

}

bool GLTexture2D::get( Array2DWriteView< Vector3f > output,
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

bool GLTexture2D::get( Array2DWriteView< Vector4f > output,
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

bool GLTexture2D::set( Array2DReadView< uint8_t > srcData,
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

bool GLTexture2D::set( Array2DReadView< uint8x2 > srcData,
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

bool GLTexture2D::set( Array2DReadView< uint8x3 > srcData,
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

bool GLTexture2D::set( Array2DReadView< uint8x4 > srcData,
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

bool GLTexture2D::set( Array2DReadView< float > srcData,
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

bool GLTexture2D::set( Array2DReadView< Vector2f > srcData,
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

bool GLTexture2D::set( Array2DReadView< Vector3f > srcData,
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

bool GLTexture2D::set( Array2DReadView< Vector4f > srcData,
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

void GLTexture2D::drawNV( GLSamplerObject* sampler,
                         float z,
                         const Rect2f& texCoords )
{
    drawNV( Rect2f( size() ), sampler, z, texCoords );
}

void GLTexture2D::drawNV( const Rect2f& windowCoords,
                         GLSamplerObject* sampler,
                         float z,
                         const Rect2f& texCoords )
{
    GLuint samplerId = sampler == nullptr ? 0 : sampler->id();

    Vector2f p0 = windowCoords.leftBottom();
    Vector2f p1 = windowCoords.rightTop();

    Vector2f t0 = texCoords.leftBottom();
    Vector2f t1 = texCoords.rightTop();

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
