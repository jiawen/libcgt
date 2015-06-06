#include "GLTexture.h"

#include <cstdio>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

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

void GLTexture::bind( GLenum texunit )
{
    // TODO: ARB_DSA: glBindTextureUnit: takes an uint, not an enum!
    glBindMultiTextureEXT( texunit, m_target, m_id );
}

void GLTexture::unbind( GLenum texunit )
{
    glBindMultiTextureEXT( texunit, m_target, 0 );
}

GLuint GLTexture::id() const
{
	return m_id;
}

GLenum GLTexture::target() const
{
	return m_target;
}

GLImageInternalFormat GLTexture::internalFormat() const
{
	return m_internalFormat;
}

//////////////////////////////////////////////////////////////////////////
// Protected
//////////////////////////////////////////////////////////////////////////

GLTexture::GLTexture( GLenum target, GLImageInternalFormat internalFormat ) :
    m_target( target ),
	m_internalFormat( internalFormat )
{
	glGenTextures( 1, &m_id );
}
