#include "GLTexture.h"

#include <cstdio>

#include "libcgt/core/math/Arithmetic.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

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
    destroy();
}

void GLTexture::bind( GLuint textureUnitIndex ) const
{
    glBindTextureUnit( textureUnitIndex, m_id );
}

void GLTexture::unbind( GLuint textureUnitIndex ) const
{
    glBindTextureUnit( textureUnitIndex, 0 );
}

void GLTexture::clear( uint8_t clearValue, GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_UNSIGNED_BYTE, &clearValue );
}

void GLTexture::clear( const uint8x2& clearValue,
    GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_UNSIGNED_BYTE, &clearValue );
}

void GLTexture::clear( const uint8x3& clearValue,
    GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_UNSIGNED_BYTE, &clearValue );
}

void GLTexture::clear( const uint8x4& clearValue,
    GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_UNSIGNED_BYTE, &clearValue );
}

void GLTexture::clear( float clearValue, GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_FLOAT, &clearValue );
}

void GLTexture::clear( const Vector2f& clearValue,
    GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_FLOAT, &clearValue );
}

void GLTexture::clear( const Vector3f& clearValue,
    GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_FLOAT, &clearValue );
}

void GLTexture::clear( const Vector4f& clearValue,
    GLImageFormat srcFormat, int level )
{
    glClearTexImage( id(), level, glImageFormat( srcFormat ),
        GL_FLOAT, &clearValue );
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

void GLTexture::setSwizzleRGB( GLTexture::SwizzleTarget rgbTarget )
{
    setSwizzle( GLTexture::SwizzleSource::RED, rgbTarget );
    setSwizzle( GLTexture::SwizzleSource::GREEN, rgbTarget );
    setSwizzle( GLTexture::SwizzleSource::BLUE, rgbTarget );
}

void GLTexture::setSwizzle( GLTexture::SwizzleSource source,
    GLTexture::SwizzleTarget target )
{
    glTextureParameteri( id(), static_cast< GLenum >( source ),
        static_cast< GLint >( target ) );
}

void GLTexture::setSwizzleRGBA( GLTexture::SwizzleTarget target )
{
    GLTexture::SwizzleTarget glTargets[ 4 ] =
        { target, target, target, target };
    setSwizzleRGBA( glTargets );
}

void GLTexture::setSwizzleRGBAlpha( GLTexture::SwizzleTarget rgbTarget,
    GLTexture::SwizzleTarget alphaTarget )
{
    GLTexture::SwizzleTarget glTargets[ 4 ] =
        { rgbTarget, rgbTarget, rgbTarget, alphaTarget };
    setSwizzleRGBA( glTargets );
}

void GLTexture::setSwizzleRGBA( GLTexture::SwizzleTarget targets[ 4 ] )
{
    GLint* glTargets = reinterpret_cast< GLint* >( targets );
    glTextureParameteriv( id(), GL_TEXTURE_SWIZZLE_RGBA, glTargets );
}

void GLTexture::setSwizzleAlpha( SwizzleTarget target )
{
    setSwizzle( GLTexture::SwizzleSource::ALPHA, target );
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

GLTexture::GLTexture( GLTexture&& move )
{
    destroy();
    m_id = move.m_id;
    m_target = move.m_target;
    m_internalFormat = move.m_internalFormat;
    m_nMipMapLevels = move.m_nMipMapLevels;

    move.m_id = 0;
    move.m_nMipMapLevels = 0;
}

GLTexture& GLTexture::operator = ( GLTexture&& move )
{
    if( this != &move )
    {
        destroy();
        m_id = move.m_id;
        m_target = move.m_target;
        m_internalFormat = move.m_internalFormat;
        m_nMipMapLevels = move.m_nMipMapLevels;

        move.m_id = 0;
        move.m_nMipMapLevels = 0;
    }
    return *this;
}

void GLTexture::destroy()
{
    if( m_id != 0 )
    {
        glDeleteTextures( 1, &m_id );
        m_id = 0;
        m_nMipMapLevels = 0;
    }
}
