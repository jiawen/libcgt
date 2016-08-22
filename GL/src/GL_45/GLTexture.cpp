#include "GLTexture.h"

#include <cstdio>

#include <math/Arithmetic.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

using libcgt::core::math::log2;

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
int GLTexture::maxSize1D()
{
    int maxSize;
    glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxSize );
    return maxSize;
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

void GLTexture::bind( GLuint textureUnitIndex ) const
{
    glBindTextureUnit( textureUnitIndex, m_id );
}

void GLTexture::unbind( GLuint textureUnitIndex ) const
{
    glBindTextureUnit( textureUnitIndex, 0 );
}

void GLTexture::clear( const uint8x4& clearValue, int level )
{
    glClearTexImage( id(), level, GL_RGBA, GL_UNSIGNED_BYTE, &clearValue );
}

void GLTexture::clear( float clearValue, GLImageFormat format, int level )
{
    glClearTexImage( id(), level, glImageFormat( format ), GL_FLOAT,
                    &clearValue );
}

void GLTexture::clear( const Vector2f& clearValue, int level )
{
    glClearTexImage( id(), level, GL_RG, GL_FLOAT, &clearValue );
}

void GLTexture::clear( const Vector3f& clearValue, int level )
{
    glClearTexImage( id(), level, GL_RGB, GL_FLOAT, &clearValue );
}

void GLTexture::clear( const Vector4f& clearValue, int level )
{
    glClearTexImage( id(), level, GL_RGBA, GL_FLOAT, &clearValue );
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

void GLTexture::setSwizzle( GLTexture::SwizzleSource source,
    GLTexture::SwizzleTarget target )
{
    glTextureParameteri( id(), static_cast< GLenum >( source ),
        static_cast< GLint >( target ) );
}

void GLTexture::setSwizzleRGBA( GLTexture::SwizzleTarget targets[ 4 ] )
{
    GLint* glTargets = reinterpret_cast< GLint* >( targets );
    glTextureParameteriv( id(), GL_TEXTURE_SWIZZLE_RGBA, glTargets );
}

void GLTexture::generateMipMaps()
{
    glGenerateTextureMipmap( id() );
}

GLTexture::GLTexture( Target target, GLImageInternalFormat internalFormat,
                     GLsizei nMipMapLevels ) :
    m_id( 0 ),
    m_target( target ),
    m_internalFormat( internalFormat )
{
    if( nMipMapLevels == 0 )
    {
        m_nMipMapLevels = log2( nMipMapLevels );
    }
    else
    {
        m_nMipMapLevels = nMipMapLevels;
    }
    glCreateTextures( glTarget(), 1, &m_id );
}
