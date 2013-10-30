#include <cassert>
#include <cstdio>

#include <math/MathUtils.h>

#include "GLTexture1D.h"
#include "GLUtilities.h"

// ========================================
// Public
// ========================================

// static
void GLTexture1D::setEnabled( bool bEnabled )
{
	if( bEnabled )
	{
		glEnable( GL_TEXTURE_1D );
	}
	else
	{
		glDisable( GL_TEXTURE_1D );
	}
}

// static
GLTexture1D* GLTexture1D::createUnsignedByte1Texture( int width, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::R_8_BYTE_UNORM;

	GLTexture1D* pTexture = GLTexture1D::createTexture1D( width, internalFormat );

	glTexImage1D( GL_TEXTURE_1D, 0, internalFormat, width, 0, GL_RED, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture1D* GLTexture1D::createUnsignedByte3Texture( int width, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGB_8_BYTE_UNORM;

	GLTexture1D* pTexture = GLTexture1D::createTexture1D( width, internalFormat );

	glTexImage1D( GL_TEXTURE_1D, 0, internalFormat, width, 0, GL_RGB, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture1D* GLTexture1D::createUnsignedByte4Texture( int width, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGBA_8_BYTE_UNORM;

	GLTexture1D* pTexture = GLTexture1D::createTexture1D( width, internalFormat );

	glTexImage1D( GL_TEXTURE_1D, 0, internalFormat, width, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture1D* GLTexture1D::createFloat1Texture( int width, int nBitsPerComponent, const float* data )
{
	GLTexture::GLTextureInternalFormat internalFormat;

	switch( nBitsPerComponent )
	{
	case 16:
		internalFormat = GLTexture::R_16_FLOAT;
		break;
	case 32:
		internalFormat = GLTexture::R_32_FLOAT;
		break;
	default:
		fprintf( stderr, "Floating point texture nBits must be 16 or 32 bits!\n" );
		assert( false );
	}

	GLTexture1D* pTexture = GLTexture1D::createTexture1D( width, internalFormat );

	glTexImage1D( GL_TEXTURE_1D, 0, internalFormat, width, 0, GL_RED, GL_FLOAT, data );

	return pTexture;
}

// static
GLTexture1D* GLTexture1D::createFloat3Texture( int width, int nBits, const float* data )
{
	GLTexture::GLTextureInternalFormat internalFormat;

	switch( nBits )
	{
	case 16:
		internalFormat = GLTexture::RGB_16_FLOAT;
		break;
	case 32:
		internalFormat = GLTexture::RGB_32_FLOAT;
		break;
	default:
		fprintf( stderr, "Floating point texture nBits must be 16 or 32 bits!\n" );
		assert( false );
	}

	GLTexture1D* pTexture = GLTexture1D::createTexture1D( width, internalFormat );

	glTexImage1D( GL_TEXTURE_1D, 0, internalFormat, width, 0, GL_RGB, GL_FLOAT, data );

	return pTexture;
}

// static
GLTexture1D* GLTexture1D::createFloat4Texture( int width, int nBits, const float* data )
{
	GLTexture::GLTextureInternalFormat internalFormat;

	switch( nBits )
	{
	case 16:
		internalFormat = GLTexture::RGBA_16_FLOAT;
		break;
	case 32:
		internalFormat = GLTexture::RGBA_32_FLOAT;
		break;
	default:
		fprintf( stderr, "Floating point texture nBits must be 16 or 32 bits!\n" );
		assert( false );
	}

	GLTexture1D* pTexture = GLTexture1D::createTexture1D( width, internalFormat );

	glTexImage1D( GL_TEXTURE_1D, 0, internalFormat, width, 0, GL_RGBA, GL_FLOAT, data );

	return pTexture;
}

void GLTexture1D::setFloat1Data( const float* data, int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

    // TODO: glTexSubImage
	glTexImage1D( GL_TEXTURE_1D, 0, getInternalFormat(), width, 0, GL_RED, GL_FLOAT, data );
}

void GLTexture1D::setFloat3Data( const float* data, int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, getInternalFormat(), width, 0, GL_RGB, GL_FLOAT, data );
}

void GLTexture1D::setFloat4Data( const float* data, int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, getInternalFormat(), width, 0, GL_RGBA, GL_FLOAT, data );
}

void GLTexture1D::setUnsignedByte1Data( const uint8_t* data,
									   int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, getInternalFormat(), width, 0, GL_RED, GL_UNSIGNED_BYTE, data );
}

void GLTexture1D::setUnsignedByte3Data( const uint8_t* data,
									   int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, getInternalFormat(), width, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
}

void GLTexture1D::setUnsignedByte4Data( const uint8_t* data,
									   int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, getInternalFormat(), width, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
}

int GLTexture1D::getWidth()
{
	return m_width;
}

void GLTexture1D::setAllWrapModes( GLint iMode )
{
	bind();
	glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, iMode );
	glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, iMode );
}

// virtual
uint8_t* GLTexture1D::getUByteData( GLint level )
{
	// TODO: set for other levels
	assert( level == 0 );

	bind();

	GLenum format;
	int nComponents = getNumComponents();
	switch( nComponents )
	{
	case 1:
		format = GL_LUMINANCE;
		break;
	case 3:
		format = GL_RGB;
		break;
	case 4:
		format = GL_RGBA;
		break;
	}

	uint8_t* pixels = new ubyte[ nComponents * m_width ];
	glGetTexImage( GL_TEXTURE_1D, level, format, GL_UNSIGNED_BYTE, pixels );
	return pixels;
}

// virtual
float* GLTexture1D::getFloatData( GLint level )
{
	// TODO: set for other levels
	assert( level == 0 );

	bind();

	GLenum format;
	int nComponents = getNumComponents();
	switch( nComponents )
	{
	case 1:
		format = GL_RED;
		break;
	case 3:
		format = GL_RGB;
		break;
	case 4:
		format = GL_RGBA;
		break;
	}

	float* pixels = new float[ nComponents * m_width ];
	glGetTexImage( GL_TEXTURE_1D, level, format, GL_FLOAT, pixels );
	return pixels;
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
GLTexture1D* GLTexture1D::createTexture1D( int width, GLTexture::GLTextureInternalFormat internalFormat )
{
	GLTexture1D* pTexture = new GLTexture1D( width, internalFormat );
	pTexture->setAllWrapModes( GL_CLAMP );
	pTexture->setFilterMode( GLTexture::GLTextureFilterMode_NEAREST, GLTexture::GLTextureFilterMode_NEAREST );
	return pTexture;
}

GLTexture1D::GLTexture1D( int width, GLTexture::GLTextureInternalFormat internalFormat ) :

    GLTexture( GL_TEXTURE_1D, internalFormat ),
	m_width( width )
{
	assert( width > 0 );
}
