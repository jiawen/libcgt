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

    glTextureStorage2DEXT( id(), GL_TEXTURE_RECTANGLE, 1, static_cast< GLenum >( internalFormat ), size.x, size.y );
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

void GLTextureRectangle::set( Array2DView< const uint8_t > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id(), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat() ), m_size.x, m_size.y, 0, GL_RED, GL_UNSIGNED_BYTE, data );
    }
}

void GLTextureRectangle::set( Array2DView< const uint8x2 > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id( ), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat( ) ), m_size.x, m_size.y, 0, GL_RG, GL_UNSIGNED_BYTE, data );
    }
}

void GLTextureRectangle::set( Array2DView< const uint8x3 > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id( ), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat( ) ), m_size.x, m_size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
    }
}

void GLTextureRectangle::set( Array2DView< const uint8x4 > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id( ), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat( ) ), m_size.x, m_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
    }
}

void GLTextureRectangle::set( Array2DView< const float > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id( ), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat( ) ), m_size.x, m_size.y, 0, GL_RED, GL_FLOAT, data );

    }
}

void GLTextureRectangle::set( Array2DView< const Vector2f > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id( ), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat( ) ), m_size.x, m_size.y, 0, GL_RG, GL_FLOAT, data );
    }
}

void GLTextureRectangle::set( Array2DView< const Vector3f > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id( ), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat( ) ), m_size.x, m_size.y, 0, GL_RGB, GL_FLOAT, data );
    }
}

void GLTextureRectangle::set( Array2DView< const Vector4f > data )
{
    if( data.packed() )
    {
        glTextureImage2DEXT( id( ), GL_TEXTURE_RECTANGLE, 0, static_cast< GLenum >( internalFormat( ) ), m_size.x, m_size.y, 0, GL_RGBA, GL_FLOAT, data );
    }
}
