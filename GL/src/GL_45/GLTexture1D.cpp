#include "GLTexture1D.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#include <math/Arithmetic.h>

#include "GLUtilities.h"

#include <algorithm>

// static
int GLTexture1D::calculateNumMipMapLevels( int size )
{
    return 1 + Arithmetic::log2( size );
}

// static
int GLTexture1D::calculateMipMapSizeForLevel( int baseSize, int level )
{
    if( level <= 0 )
    {
        return baseSize;
    }

    int size = baseSize;
    while( level > 0 )
    {
        size = std::max( 1, size / 2 );
        --level;
    }
    return size;
}

GLTexture1D::GLTexture1D( int width, GLImageInternalFormat internalFormat,
                         GLsizei nMipMapLevels ) :
    GLTexture( GLTexture::Target::TEXTURE_1D, internalFormat, nMipMapLevels ),
    m_width( width )
{
    assert( width > 0 );
    assert( width <= GLTexture::maxSize1D() );
    glTextureStorage1D( id(), 1, static_cast< GLenum >( internalFormat ),
                       width );
}

int GLTexture1D::numElements() const
{
    return m_width;
}

int GLTexture1D::width() const
{
    return m_width;
}

int GLTexture1D::size() const
{
    return m_width;
}

bool GLTexture1D::set( Array1DView< const uint8_t > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
{
    if( srcFormat != GLImageFormat::RED &&
       srcFormat != GLImageFormat::GREEN &&
       srcFormat != GLImageFormat::BLUE &&
       srcFormat != GLImageFormat::ALPHA )
    {
        return false;
    }
    if( srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::set( Array1DView< const uint8x2 > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
{
    if( srcFormat != GLImageFormat::RG )
    {
        return false;
    }
    if( srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::UNSIGNED_BYTE;
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::set( Array1DView< const uint8x3 > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
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
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::set( Array1DView< const uint8x4 > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
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
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::set( Array1DView< const float > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
{
    if( srcFormat != GLImageFormat::RED &&
       srcFormat != GLImageFormat::GREEN &&
       srcFormat != GLImageFormat::BLUE &&
       srcFormat != GLImageFormat::ALPHA )
    {
        return false;
    }
    if( srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::set( Array1DView< const Vector2f > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
{
    if( srcFormat != GLImageFormat::RG )
    {
        return false;
    }
    if( srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::set( Array1DView< const Vector3f > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
{
    if( srcFormat != GLImageFormat::RGB && srcFormat != GLImageFormat::BGR )
    {
        return false;
    }
    if( srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::set( Array1DView< const Vector4f > srcData,
                      GLImageFormat srcFormat,
                      int dstOffset )
{
    if( srcFormat != GLImageFormat::RGBA && srcFormat != GLImageFormat::BGRA )
    {
        return false;
    }
    if( srcData.packed() )
    {
        return false;
    }
    const GLPixelType srcType = GLPixelType::FLOAT;
    return set1D( srcData.pointer(), srcData.size(), srcFormat, srcType,
                 dstOffset );
}

bool GLTexture1D::checkSize( int srcSize, int dstOffset ) const
{
    if( dstOffset + srcSize > m_width )
    {
        return false;
    }
    return true;
}

bool GLTexture1D::get1D( GLint srcLevel,
                        GLImageFormat dstFormat, GLPixelType dstType,
                        int dstSize, void* dstPtr )
{
    if( dstPtr == nullptr )
    {
        return false;
    }
    // TODO: rename checkSize to support srcLevel properly.
    bool succeeded = checkSize( dstSize, 0 );
    if( succeeded )
    {
        glGetTextureImage
        (
            id(), srcLevel,
            glImageFormat( dstFormat ), glPixelType( dstType ),
            dstSize,
            dstPtr
        );
        return true;
    }
    return succeeded;
}

bool GLTexture1D::set1D( const void* srcPtr, int srcSize,
                        GLImageFormat srcFormat, GLPixelType srcType,
                        int dstOffset )
{
    if( srcPtr == nullptr )
    {
        return false;
    }
    bool succeeded = checkSize( srcSize, dstOffset );
    if( succeeded )
    {
        glTextureSubImage1D
        (
            id(), 0, // TODO: level
            dstOffset, srcSize,
            glImageFormat( srcFormat ), glPixelType( srcType ),
            srcPtr
        );
    }
    return succeeded;
}