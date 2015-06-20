#include "GLTexture3D.h"

#include <cassert>
#include <cstdio>
#include <vector>

#include "GLUtilities.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

GLTexture3D::GLTexture3D( const Vector3i& size, GLImageInternalFormat internalFormat ) :

    GLTexture( GL_TEXTURE_3D, internalFormat ),
    m_size( size )
{
    assert( m_size.x > 0 );
    assert( m_size.y > 0 );
    assert( m_size.z > 0 );
    assert( m_size.x <= GLTexture::maxSize3D() );
    assert( m_size.y <= GLTexture::maxSize3D( ) );
    assert( m_size.z <= GLTexture::maxSize3D( ) );

    glTextureStorage3D( id(), 1, static_cast< GLenum >( internalFormat ),
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

void GLTexture3D::setFloat1Data( const float* data, int xOffset, int yOffset, int zOffset, int width, int height, int depth )
{
    bind();

    if( width == 0 )
    {
        width = m_size.x;
    }
    if( height == 0 )
    {
        height = m_size.y;
    }
    if( depth == 0 )
    {
        depth = m_size.z;
    }

    glTexImage3D( GL_TEXTURE_3D, 0, static_cast< GLint >( internalFormat() ), width, height, depth, 0, GL_RED, GL_FLOAT, data );
}

void GLTexture3D::setFloat3Data( const float* data, int xOffset, int yOffset, int zOffset, int width, int height, int depth )
{
    bind();

    if( width == 0 )
    {
        width = m_size.x;
    }
    if( height == 0 )
    {
        height = m_size.y;
    }
    if( depth == 0 )
    {
        depth = m_size.z;
    }

    glTexImage3D( GL_TEXTURE_3D, 0, static_cast< GLint >( internalFormat() ), width, height, depth, 0, GL_RGB, GL_FLOAT, data );
}

void GLTexture3D::setFloat4Data( const float* data, int xOffset, int yOffset, int zOffset, int width, int height, int depth )
{
    bind();

    if( width == 0 )
    {
        width = m_size.x;
    }
    if( height == 0 )
    {
        height = m_size.y;
    }
    if( depth == 0 )
    {
        depth = m_size.z;
    }

    glTexImage3D( GL_TEXTURE_3D, 0, static_cast< GLint >( internalFormat() ), width, height, depth, 0, GL_RGBA, GL_FLOAT, data );
}

void GLTexture3D::setUnsignedByte1Data( const uint8_t* data,
                                       int xOffset, int yOffset, int zOffset,
                                       int width, int height, int depth )
{
    bind();

    if( width == 0 )
    {
        width = m_size.x;
    }
    if( height == 0 )
    {
        height = m_size.y;
    }
    if( depth == 0 )
    {
        depth = m_size.z;
    }

    glTexImage3D( GL_TEXTURE_3D, 0, static_cast< GLint >( internalFormat() ), width, height, depth, 0, GL_RED, GL_UNSIGNED_BYTE, data );
}

void GLTexture3D::setUnsignedByte3Data( const uint8_t* data,
                                       int xOffset, int yOffset, int zOffset,
                                       int width, int height, int depth )
{
    bind();

    if( width == 0 )
    {
        width = m_size.x;
    }
    if( height == 0 )
    {
        height = m_size.y;
    }
    if( depth == 0 )
    {
        depth = m_size.z;
    }

    glTexImage3D( GL_TEXTURE_3D, 0, static_cast< GLint >( internalFormat() ), width, height, depth, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
}

void GLTexture3D::setUnsignedByte4Data( const uint8_t* data,
                                       int xOffset, int yOffset, int zOffset,
                                       int width, int height, int depth )
{
    bind();

    if( width == 0 )
    {
        width = m_size.x;
    }
    if( height == 0 )
    {
        height = m_size.y;
    }
    if( depth == 0 )
    {
        depth = m_size.z;
    }

    glTexImage3D( GL_TEXTURE_3D, 0, static_cast< GLint >( internalFormat() ), width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
}
