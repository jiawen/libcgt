#include "libcgt/GL/GL_45/GLTextureRectangle.h"

#include <cassert>
#include <cstdio>

#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"
#include "libcgt/GL/GLUtilities.h"

// TODO: support mip maps
GLTextureRectangle::GLTextureRectangle( const Vector2i& size, GLImageInternalFormat internalFormat ) :
    GLTexture( GLTexture::Target::TEXTURE_RECTANGLE, internalFormat, 1 ),
    m_size( size )
{
    assert( size.x > 0 );
    assert( size.y > 0 );
    assert( size.x <= GLTexture::maxSize2D() );
    assert( size.y <= GLTexture::maxSize2D() );

    glTextureStorage2D( id(), 1, static_cast< GLenum >( internalFormat ), size.x, size.y );
}

int GLTextureRectangle::numElements() const
{
    return m_size.x * m_size.y;
}

int GLTextureRectangle::width() const
{
    return m_size.x;
}

int GLTextureRectangle::height() const
{
    return m_size.y;
}

Vector2i GLTextureRectangle::size() const
{
    return m_size;
}

bool GLTextureRectangle::set( Array2DReadView< uint8_t > srcData,
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

bool GLTextureRectangle::set( Array2DReadView< uint8x2 > srcData,
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

bool GLTextureRectangle::set( Array2DReadView< uint8x3 > srcData,
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

bool GLTextureRectangle::set( Array2DReadView< uint8x4 > srcData,
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

bool GLTextureRectangle::set( Array2DReadView< float > srcData,
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

bool GLTextureRectangle::set( Array2DReadView< Vector2f > srcData,
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

bool GLTextureRectangle::set( Array2DReadView< Vector3f > srcData,
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

bool GLTextureRectangle::set( Array2DReadView< Vector4f > srcData,
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

bool GLTextureRectangle::checkSize( const Vector2i& srcSize,
                                   const Vector2i& dstOffset ) const
{
    if( dstOffset.x + srcSize.x > m_size.x ||
        dstOffset.y + srcSize.y > m_size.y )
    {
        return false;
    }
    return true;
}

bool GLTextureRectangle::set2D( const void* srcPtr, const Vector2i& srcSize,
                               GLImageFormat srcFormat, GLPixelType srcType,
                               const Vector2i& dstOffset )
{
    bool succeeded = checkSize( srcSize, dstOffset );
    if( succeeded )
    {
        glTextureSubImage2D
        (
            id(), 0,
            dstOffset.x, dstOffset.y, srcSize.x, srcSize.y,
            glImageFormat( srcFormat ), glPixelType( srcType ),
            srcPtr
        );
    }
    return succeeded;
}
