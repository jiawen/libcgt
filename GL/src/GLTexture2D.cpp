#include <cassert>
#include <cstdio>

#include <cstdint>

#include <common/Array2DView.h>
#include <GL/glew.h>
#include <math/MathUtils.h>
#include <QFile>
#include <QImage>

#include <io/PortableFloatMapIO.h>
#include <io/PortablePixelMapIO.h>

#include <vecmath/Vector3f.h>

#include "GLTexture2D.h"
#include "GLUtilities.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
void GLTexture2D::setEnabled( bool bEnabled )
{
	if( bEnabled )
	{
		glEnable( GL_TEXTURE_2D );
	}
	else
	{
		glDisable( GL_TEXTURE_2D );
	}
}

// static
GLTexture2D* GLTexture2D::createDepthTexture( int width, int height, int nBits, const uint* data )
{
	GLTexture::GLTextureInternalFormat internalFormat;

	switch( nBits )
	{
	case 16:
		internalFormat = GLTexture::DEPTH_COMPONENT_16;
		break;
	case 24:
		internalFormat = GLTexture::DEPTH_COMPONENT_24;
		break;
	case 32:
		internalFormat = GLTexture::DEPTH_COMPONENT_32;
		break;
	default:
		fprintf( stderr, "Depth texture precision must be 16, 24, or 32 bits" );
		assert( false );
		break;
	}
    
    // TODO: float32 creator, GL_DEPTH24_STENCIL8 creator

	GLTexture2D* pTexture = GLTexture2D::createTexture2D( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, data );

	return pTexture;
}

// static
GLTexture2D* GLTexture2D::createUnsignedByte1Texture( int width, int height, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::R_8_BYTE_UNORM;

	GLTexture2D* pTexture = GLTexture2D::createTexture2D( width, height,
		internalFormat );

	glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture2D* GLTexture2D::createUnsignedByte3Texture( int width, int height, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGB_8_BYTE_UNORM;

	GLTexture2D* pTexture = GLTexture2D::createTexture2D( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture2D* GLTexture2D::createUnsignedByte4Texture( int width, int height, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGBA_8_BYTE_UNORM;

	GLTexture2D* pTexture = GLTexture2D::createTexture2D( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture2D* GLTexture2D::createFloat1Texture( int width, int height, int nBitsPerComponent, const float* data )
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

	GLTexture2D* pTexture = GLTexture2D::createTexture2D( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_RED, GL_FLOAT, data );

	return pTexture;
}

// static
GLTexture2D* GLTexture2D::createFloat3Texture( int width, int height, int nBits, const float* data )
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

	GLTexture2D* pTexture = GLTexture2D::createTexture2D( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_RGB, GL_FLOAT, data );

	return pTexture;
}

// static
GLTexture2D* GLTexture2D::createFloat4Texture( int width, int height, int nBits, const float* data )
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

	GLTexture2D* pTexture = GLTexture2D::createTexture2D( width, height, internalFormat );
	
	glTexImage2D( GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_RGBA, GL_FLOAT, data );

	return pTexture;
}

void GLTexture2D::setFloat1Data( const float* data, int xOffset, int yOffset, int width, int height )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}
	if( height == 0 )
	{
		height = m_height;
	}

	glTexImage2D( GL_TEXTURE_2D, 0, getInternalFormat(), width, height, 0, GL_RED, GL_FLOAT, data );
}

void GLTexture2D::setFloat3Data( const float* data, int xOffset, int yOffset, int width, int height )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}
	if( height == 0 )
	{
		height = m_height;
	}

	glTexImage2D( GL_TEXTURE_2D, 0, getInternalFormat(), width, height, 0, GL_RGB, GL_FLOAT, data );
}

void GLTexture2D::setFloat4Data( const float* data, int xOffset, int yOffset, int width, int height )
{
	bind();
	if( width == 0 )
	{
		width = m_width;
	}
	if( height == 0 )
	{
		height = m_height;
	}

	glTexImage2D( GL_TEXTURE_2D, 0, getInternalFormat(), width, height, 0, GL_RGBA, GL_FLOAT, data );
}

void GLTexture2D::setUnsignedByte1Data( const uint8_t* data,
									   int xOffset, int yOffset,
									   int width, int height )
{
	bind();

	if( width == 0 )
	{
		width = m_width;
	}
	if( height == 0 )
	{
		height = m_height;
	}

	glTexImage2D( GL_TEXTURE_2D, 0, getInternalFormat(), width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data );
}

void GLTexture2D::setUnsignedByte3Data( const uint8_t* data,
									   int xOffset, int yOffset,
									   int width, int height )
{
	bind();

	if( width == 0 )
	{
		width = m_width;
	}
	if( height == 0 )
	{
		height = m_height;
	}

	glTexImage2D( GL_TEXTURE_2D, 0, getInternalFormat(), width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
}

void GLTexture2D::setUnsignedByte4Data( const uint8_t* data,
									   int xOffset, int yOffset,
									   int width, int height )
{
	bind();

	if( width == 0 )
	{
		width = m_width;
	}
	if( height == 0 )
	{
		height = m_height;
	}

	glTexImage2D( GL_TEXTURE_2D, 0, getInternalFormat(), width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
}

int GLTexture2D::getWidth()
{
	return m_width;
}

int GLTexture2D::getHeight()
{
	return m_height;
}

void GLTexture2D::setAllWrapModes( GLint iMode )
{
	bind();
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, iMode );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, iMode );
}

// virtual
void GLTexture2D::dumpToCSV( QString filename )
{
	// TODO: use getData(), use it to get the right formats

	float* pixels = new float[ 4 * m_width * m_height ];
	getFloat4Data( pixels );

	int k = 0;
	FILE* fp = fopen( qPrintable( filename ), "w" );
	fprintf( fp, "width = %d,height = %d\n", m_width, m_height );
	fprintf( fp, "pixel,byte,x,y,r,g,b,a\n" );

	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			fprintf( fp, "%d,%d,%d,%d,%f,%f,%f,%f\n", k / 4, k, x, y, pixels[k], pixels[k+1], pixels[k+2], pixels[k+3] );
			k += 4;
		}
	}

	fclose( fp );

	delete[] pixels;
}

// virtual
void GLTexture2D::dumpToTXT( QString filename, GLint level, GLenum format, GLenum type )
{
	// TODO: use getData(), use it to get the right formats

	assert( level == 0 );
	assert( format == GL_RGBA );
	assert( type == GL_FLOAT );

	float* pixels = new float[ 4 * m_width * m_height ];
	getFloat4Data( pixels );

	int k = 0;
	FILE* fp = fopen( qPrintable( filename ), "w" );
	fprintf( fp, "width = %d, height = %d\n", m_width, m_height );

	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			fprintf( fp, "{%d} [%d] (%d, %d): <%f, %f, %f, %f>\n", k / 4, k, x, y, pixels[k], pixels[k+1], pixels[k+2], pixels[k+3] );
			k += 4;
		}
	}

	fclose( fp );

	delete[] pixels;
}

void GLTexture2D::dumpToPPM( QString filename )
{
    std::vector< uint8_t > pixels( 3 * m_width * m_height );
	getUnsignedByte3Data( pixels.data() );
    Array2DView< ubyte3 > view( pixels.data(), m_width, m_height );
	PortablePixelMapIO::writeRGB( filename, view );
}

void GLTexture2D::dumpToPNG( QString filename )
{
    std::vector< uint8_t > pixels( 4 * m_width * m_height );
	getUnsignedByte4Data( pixels.data() );
	
	// TODO: arrayToQImage
	QImage q( m_width, m_height, QImage::Format_ARGB32 );
	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			int yy = m_height - y - 1;
			int k = 4 * ( yy * m_width + x );

			uint8_t r = pixels[ k ];
			uint8_t g = pixels[ k + 1 ];
			uint8_t b = pixels[ k + 2 ];
			uint8_t a = pixels[ k + 3 ];

			q.setPixel( x, y, qRgba( r, g, b, a ) );
		}
	}
	q.save( filename, "PNG" );
}

void GLTexture2D::dumpToPFM( QString filename )
{
    std::vector< float > pixels( 3 * m_width * m_height );
	getFloat3Data( pixels.data() );
    Array2DView< Vector3f > view( pixels.data(), m_width, m_height );
	PortableFloatMapIO::writeRGB( filename, view );
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
GLTexture2D* GLTexture2D::createTexture2D( int width, int height,
										  GLTexture::GLTextureInternalFormat internalFormat )
{
	GLTexture2D* pTexture = new GLTexture2D( width, height, internalFormat );
	pTexture->setAllWrapModes( GL_CLAMP );
	pTexture->setFilterMode( GLTexture::GLTextureFilterMode_NEAREST, GLTexture::GLTextureFilterMode_NEAREST );
	return pTexture;
}

GLTexture2D::GLTexture2D( int width, int height,
						 GLTexture::GLTextureInternalFormat internalFormat ) :

    GLTexture( GL_TEXTURE_2D, internalFormat ),

	m_width( width ),
	m_height( height )
{
	assert( width > 0 );
	assert( height > 0 );
}
