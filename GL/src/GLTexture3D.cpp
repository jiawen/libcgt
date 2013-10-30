#include <cassert>
#include <cstdio>

#include <GL/glew.h>

#include <QImage>

#include "GLTexture3D.h"
#include "GLUtilities.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
void GLTexture3D::setEnabled( bool bEnabled )
{
	if( bEnabled )
	{
		glEnable( GL_TEXTURE_3D );
	}
	else
	{
		glDisable( GL_TEXTURE_3D );
	}
}

// static
GLTexture3D* GLTexture3D::createUnsignedByte1Texture( int width, int height, int depth, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::R_8_BYTE_UNORM;

	GLTexture3D* pTexture = GLTexture3D::createTexture3D( width, height, depth, internalFormat );

	glTexImage3D( GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, GL_RED, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture3D* GLTexture3D::createUnsignedByte3Texture( int width, int height, int depth, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGB_8_BYTE_UNORM;

	GLTexture3D* pTexture = GLTexture3D::createTexture3D( width, height, depth, internalFormat );

	glTexImage3D( GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, GL_RGB, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture3D* GLTexture3D::createUnsignedByte4Texture( int width, int height, int depth, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGBA_8_BYTE_UNORM;

	GLTexture3D* pTexture = GLTexture3D::createTexture3D( width, height, depth, internalFormat );

	glTexImage3D( GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTexture3D* GLTexture3D::createFloat1Texture( int width, int height, int depth, int nBitsPerComponent, const float* data )
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

	GLTexture3D* pTexture = GLTexture3D::createTexture3D( width, height, depth, internalFormat );

	glTexImage3D( GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, GL_RED, GL_FLOAT, data );

	return pTexture;
}

// static
GLTexture3D* GLTexture3D::createFloat3Texture( int width, int height, int depth, int nBits, const float* data )
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

	GLTexture3D* pTexture = GLTexture3D::createTexture3D( width, height, depth, internalFormat );

	glTexImage3D( GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, GL_RGB, GL_FLOAT, data );

	return pTexture;
}

// static
GLTexture3D* GLTexture3D::createFloat4Texture( int width, int height, int depth, int nBits, const float* data )
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

	GLTexture3D* pTexture = GLTexture3D::createTexture3D( width, height, depth, internalFormat );

	glTexImage3D( GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, GL_RGBA, GL_FLOAT, data );

	return pTexture;
}

void GLTexture3D::setFloat1Data( const float* data, int xOffset, int yOffset, int zOffset, int width, int height, int depth )
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
	if( depth == 0 )
	{
		depth = m_depth;
	}

	glTexImage3D( GL_TEXTURE_3D, 0, getInternalFormat(), width, height, depth, 0, GL_RED, GL_FLOAT, data );
}

void GLTexture3D::setFloat3Data( const float* data, int xOffset, int yOffset, int zOffset, int width, int height, int depth )
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
	if( depth == 0 )
	{
		depth = m_depth;
	}

	glTexImage3D( GL_TEXTURE_3D, 0, getInternalFormat(), width, height, depth, 0, GL_RGB, GL_FLOAT, data );
}

void GLTexture3D::setFloat4Data( const float* data, int xOffset, int yOffset, int zOffset, int width, int height, int depth )
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
	if( depth == 0 )
	{
		depth = m_depth;
	}

	glTexImage3D( GL_TEXTURE_3D, 0, getInternalFormat(), width, height, depth, 0, GL_RGBA, GL_FLOAT, data );
}

void GLTexture3D::setUnsignedByte1Data( const uint8_t* data,
									   int xOffset, int yOffset, int zOffset,
									   int width, int height, int depth )
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
	if( depth == 0 )
	{
		depth = m_depth;
	}

	glTexImage3D( GL_TEXTURE_3D, 0, getInternalFormat(), width, height, depth, 0, GL_RED, GL_UNSIGNED_BYTE, data );
}

void GLTexture3D::setUnsignedByte3Data( const uint8_t* data,
									   int xOffset, int yOffset, int zOffset,
									   int width, int height, int depth )
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
	if( depth == 0 )
	{
		depth = m_depth;
	}

	glTexImage3D( GL_TEXTURE_3D, 0, getInternalFormat(), width, height, depth, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
}

void GLTexture3D::setUnsignedByte4Data( const uint8_t* data,
									   int xOffset, int yOffset, int zOffset,
									   int width, int height, int depth )
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
	if( depth == 0 )
	{
		depth = m_depth;
	}

	glTexImage3D( GL_TEXTURE_3D, 0, getInternalFormat(), width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
}

int GLTexture3D::getWidth()
{
	return m_width;
}

int GLTexture3D::getHeight()
{
	return m_height;
}

int GLTexture3D::getDepth()
{
	return m_depth;
}

// virtual
void GLTexture3D::setAllWrapModes( GLint iMode )
{
	bind();
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, iMode );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, iMode );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, iMode );
}

// virtual
void GLTexture3D::dumpToCSV( QString filename )
{
	assert( false );
}

// virtual
void GLTexture3D::dumpToTXT( QString filename, GLint level, GLenum format, GLenum type )
{
    vector< float > pixels( 4 * m_width * m_height * m_depth );
	getFloat4Data( pixels.data() );

	FILE* fp = fopen( qPrintable( filename ), "w" );
	fprintf( fp, "width = %d, height = %d, depth = %d\n", m_width, m_height, m_depth );

	for( int z = 0; z < m_depth; ++z )
	{
		for( int y = 0; y < m_height; ++y )
		{
			for( int x = 0; x < m_width; ++x )
			{
				int k = 4 * ( z * m_width * m_height + y * m_width + x );

				fprintf( fp, "{%d} [%d] (%d, %d, %d): <%f, %f, %f, %f>\n",
                        k / 4, k, x, y, z, pixels[k], pixels[k+1], pixels[k+2], pixels[k+3] );
				k += 4;
			}
		}
	}
	fclose( fp );
}

void GLTexture3D::dumpToPNG( QString filename )
{
	vector< uint8_t > pixels( 4 * m_width * m_height * m_depth );
	getUnsignedByte4Data( pixels.data() );

	QImage q( m_width * m_depth, m_height, QImage::Format_ARGB32 );	

	// tile across in x
	for( int z = 0; z < m_depth; ++z )
	{
		for( int y = 0; y < m_height; ++y )
		{
			for( int x = 0; x < m_width; ++x )
			{
				int yy = m_height - y - 1;
				int k = 4 * ( z * m_width * m_height + y * m_width + x );

				uint8_t r = pixels[ k ];
				uint8_t g = pixels[ k + 1 ];
				uint8_t b = pixels[ k + 2 ];
				uint8_t a = pixels[ k + 3 ];

				q.setPixel( z * m_width + x, yy, qRgba( r, g, b, a ) );
			}
		}
	}
	q.save( filename, "PNG" );
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
GLTexture3D* GLTexture3D::createTexture3D( int width, int height, int depth,
										  GLTexture::GLTextureInternalFormat internalFormat )
{
	GLTexture3D* pTexture = new GLTexture3D( width, height, depth, internalFormat );
	pTexture->setAllWrapModes( GL_CLAMP );
	pTexture->setFilterMode( GLTexture::GLTextureFilterMode_NEAREST, GLTexture::GLTextureFilterMode_NEAREST );
	return pTexture;
}

GLTexture3D::GLTexture3D( int width, int height, int depth,
						 GLTexture::GLTextureInternalFormat internalFormat ) :

	GLTexture( GL_TEXTURE_3D, internalFormat ),

    m_width( width ),
    m_height( height ),
    m_depth( depth )

{
	assert( width > 0 );
	assert( height > 0 );
	assert( depth >  0 );
}
