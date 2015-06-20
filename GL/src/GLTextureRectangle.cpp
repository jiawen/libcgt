#include "GLTextureRectangle.h"

#include <cassert>
#include <cstdio>

#include <GL/glew.h>

#include <math/MathUtils.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "GLUtilities.h"


//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

GLTextureRectangle::GLTextureRectangle( const Vector2i& size, GLImageInternalFormat internalFormat ) :
    GLTexture( GL_TEXTURE_RECTANGLE, internalFormat ),
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

bool GLTextureRectangle::set( Array2DView< const uint8_t > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_UNSIGNED_BYTE;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::set( Array2DView< const uint8x2 > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_UNSIGNED_BYTE;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::set( Array2DView< const uint8x3 > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_UNSIGNED_BYTE;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::set( Array2DView< const uint8x4 > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_UNSIGNED_BYTE;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::set( Array2DView< const float > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_FLOAT;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::set( Array2DView< const Vector2f > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_FLOAT;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::set( Array2DView< const Vector3f > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_FLOAT;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::set( Array2DView< const Vector4f > srcData,
    GLImageFormat srcFormat,
    const Vector2i& dstOffset )
{
    GLenum srcType = GL_FLOAT;
    if( srcData.packed() && checkSize( srcData.size(), dstOffset ) )
    {
        set2D( srcData.pointer(), srcData.size(), srcFormat, srcType, dstOffset );
        return true;
    }
    return false;
}

bool GLTextureRectangle::checkSize( const Vector2i& srcSize, const Vector2i& dstOffset )
{
    if( dstOffset.x + srcSize.x > m_size.x ||
        dstOffset.y + srcSize.y > m_size.y )
    {
        return false;
    }
    return true;
}

void GLTextureRectangle::set2D( const void* srcPtr, const Vector2i& srcSize,
    GLImageFormat srcFormat, GLenum srcType,
    const Vector2i& dstOffset )
{
    glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );

    glTextureSubImage2D
    (
        id(), 0,
        dstOffset.x, dstOffset.y, srcSize.x, srcSize.y,
        static_cast< GLenum >( srcFormat ), GL_FLOAT,
        srcPtr
    );

    glPopClientAttrib();
}