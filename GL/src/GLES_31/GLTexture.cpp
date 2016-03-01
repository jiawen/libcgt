#include "GLTexture.h"

#include <cstdio>

#include <math/Arithmetic.h>

// static
GLenum GLTexture::activeTextureUnit()
{
    int textureUnit;
    glGetIntegerv( GL_ACTIVE_TEXTURE, &textureUnit );
    return static_cast< GLenum >( textureUnit );
}

// static
int GLTexture::maxTextureImageUnits()
{
    int maxTIU;
    glGetIntegerv( GL_MAX_TEXTURE_IMAGE_UNITS, &maxTIU );
    return maxTIU;
}

// static
int GLTexture::maxCombinedTextureImageUnits()
{
    int maxCTIU;
    glGetIntegerv( GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxCTIU );
    return maxCTIU;
}

// static
int GLTexture::maxSize2D()
{
    int maxSize;
    glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxSize );
    return maxSize;
}

// static
int GLTexture::maxSize3D()
{
    int maxSize;
    glGetIntegerv( GL_MAX_3D_TEXTURE_SIZE, &maxSize );
    return maxSize;
}

// static
int GLTexture::maxSizeCubeMap()
{
    int maxSize;
    glGetIntegerv( GL_MAX_CUBE_MAP_TEXTURE_SIZE, &maxSize );
    return maxSize;
}

// static
int GLTexture::maxArrayLayers()
{
    int maxLayers;
    glGetIntegerv( GL_MAX_ARRAY_TEXTURE_LAYERS, &maxLayers );
    return maxLayers;
}

// virtual
GLTexture::~GLTexture()
{
    glDeleteTextures( 1, &m_id );
}

void GLTexture::bind() const
{
    glBindTexture( glTarget(), id() );
}

void GLTexture::unbind() const
{
    glBindTexture( glTarget(), 0 );
}

GLuint GLTexture::id() const
{
    return m_id;
}

GLTexture::Target GLTexture::target() const
{
    return m_target;
}

GLenum GLTexture::glTarget() const
{
    return static_cast< GLenum >( m_target );
}

GLImageInternalFormat GLTexture::internalFormat() const
{
    return m_internalFormat;
}

GLenum GLTexture::glInternalFormat() const
{
    return static_cast< GLenum >( m_internalFormat );
}

GLsizei GLTexture::numMipMapLevels() const
{
    return m_nMipMapLevels;
}

void GLTexture::generateMipMaps()
{
    glGenerateMipmap( glTarget() );
}

GLTexture::GLTexture( Target target, GLImageInternalFormat internalFormat,
        GLsizei nMipMapLevels ) :
    m_id( 0 ),
    m_target( target ),
    m_internalFormat( internalFormat )
{
    if( nMipMapLevels == 0 )
    {
        m_nMipMapLevels = Arithmetic::log2( nMipMapLevels );
    }
    else
    {
        m_nMipMapLevels = nMipMapLevels;
    }
    glGenTextures( 1, &m_id );
}