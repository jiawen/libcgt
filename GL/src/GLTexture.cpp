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
int GLTexture::maxSize1D2D()
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

// virtual
GLTexture::~GLTexture()
{
	unbind();
	glDeleteTextures( 1, &m_id );
}

void GLTexture::bind()
{
	glBindTexture( m_target, m_id );
}

void GLTexture::unbind()
{
	glBindTexture( m_target, 0 );
}

GLuint GLTexture::id()
{
	return m_id;
}

GLenum GLTexture::target()
{
	return m_target;
}

GLImageInternalFormat GLTexture::internalFormat()
{
	return m_internalFormat;
}

// virtual
void GLTexture::getFloat1Data( float* afOutput, int level )
{
	GLenum format = GL_RED;
	getTexImage( level, format, GL_FLOAT, afOutput );
}

// virtual
void GLTexture::getFloat3Data( float* afOutput, int level )
{
	GLenum format = GL_RGB;
	getTexImage( level, format, GL_FLOAT, afOutput );
}

// virtual
void GLTexture::getFloat4Data( float* afOutput, int level )
{
	GLenum format = GL_RGBA;
	getTexImage( level, format, GL_FLOAT, afOutput );
}

// virtual
void GLTexture::getUnsignedByte1Data( uint8_t* aubOutput, int level )
{
	GLenum format = GL_RED;
	getTexImage( level, format, GL_UNSIGNED_BYTE, aubOutput );
}

// virtual
void GLTexture::getUnsignedByte3Data( uint8_t* aubOutput, int level )
{
	GLenum format = GL_RGB;
	getTexImage( level, format, GL_UNSIGNED_BYTE, aubOutput );
}

// virtual
void GLTexture::getUnsignedByte4Data( uint8_t* aubOutput, int level )
{
	GLenum format = GL_RGBA;
	getTexImage( level, format, GL_UNSIGNED_BYTE, aubOutput );
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

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

void GLTexture::getTexImage( GLint level, GLenum format, GLenum type, void* avOutput )
{
    // TODO: don't need binding with DSA
	bind();
	glGetTexImage( m_target, level, format, type, avOutput );
}
