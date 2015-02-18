#include "GLTexture1D.h"

#include <cassert>
#include <cstdio>

#include <math/MathUtils.h>

#include "GLUtilities.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

GLTexture1D::GLTexture1D( int width, GLImageInternalFormat internalFormat ) :

    GLTexture( GL_TEXTURE_1D, internalFormat ),
	m_width( width )
{
	assert( width > 0 );
    assert( width <= GLTexture::maxSize1D() );
	glTextureStorage1DEXT( id(), GL_TEXTURE_1D, 1, static_cast< GLenum >( internalFormat ), width );
}

void GLTexture1D::setFloat1Data( const float* data, int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

    // TODO: glTexSubImage
	glTexImage1D( GL_TEXTURE_1D, 0, static_cast< GLint >( internalFormat() ), width, 0, GL_RED, GL_FLOAT, data );
}

void GLTexture1D::setFloat3Data( const float* data, int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, static_cast< GLint >( internalFormat() ), width, 0, GL_RGB, GL_FLOAT, data );
}

void GLTexture1D::setFloat4Data( const float* data, int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, static_cast< GLint >( internalFormat() ), width, 0, GL_RGBA, GL_FLOAT, data );
}

void GLTexture1D::setUnsignedByte1Data( const uint8_t* data,
									   int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, static_cast< GLint >( internalFormat() ), width, 0, GL_RED, GL_UNSIGNED_BYTE, data );
}

void GLTexture1D::setUnsignedByte3Data( const uint8_t* data,
									   int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, static_cast< GLint >( internalFormat() ), width, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
}

void GLTexture1D::setUnsignedByte4Data( const uint8_t* data,
									   int xOffset, int width )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}

	glTexImage1D( GL_TEXTURE_1D, 0, static_cast< GLint >( internalFormat() ), width, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
}

int GLTexture1D::width() const
{
	return m_width;
}
