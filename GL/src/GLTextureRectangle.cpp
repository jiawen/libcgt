#include <cassert>
#include <cstdio>
#include <memory>

#include <GL/glew.h>

#include <QImage>

#include <io/PortableFloatMapIO.h>
#include <io/PortablePixelMapIO.h>
#include <math/MathUtils.h>
#include <vecmath/Vector3f.h>

#include "GLTextureRectangle.h"
#include "GLUtilities.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
void GLTextureRectangle::setEnabled( bool bEnabled )
{
	if( bEnabled )
	{
		glEnable( GL_TEXTURE_RECTANGLE );
	}
	else
	{
		glDisable( GL_TEXTURE_RECTANGLE );
	}
}

// static
GLTextureRectangle* GLTextureRectangle::create( shared_ptr< Image4ub > image )
{
	return GLTextureRectangle::createUnsignedByte4Texture( image->size(), image->pixels() );
}

// static
GLTextureRectangle* GLTextureRectangle::create( shared_ptr< Image1f > image, int nBitsPerComponent )
{
	return GLTextureRectangle::createFloat1Texture( image->size(), nBitsPerComponent, image->pixels() );
}

// static
GLTextureRectangle* GLTextureRectangle::create( shared_ptr< Image4f > image, int nBitsPerComponent )
{
	return GLTextureRectangle::createFloat4Texture( image->size(), nBitsPerComponent, image->pixels() );
}

// static
GLTextureRectangle* GLTextureRectangle::createDepthTexture( int width, int height, int nBits, const float* data )
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
    
    // TODO: float32, depthstencil

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height, internalFormat );
	
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, data );
	
	return pTexture;
}

// static
GLTextureRectangle* GLTextureRectangle::createUnsignedByte1Texture( int width, int height, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::R_8_BYTE_UNORM;

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height,
		internalFormat );

	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

#if 0
// static
GLTextureRectangle* GLTextureRectangle::createUnsignedByte3Texture( int width, int height, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGB_8_BYTE_UNORM;

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTextureRectangle* GLTextureRectangle::createUnsignedByte3Texture( const Vector2i& size, const uint8_t* data )
{
	return createUnsignedByte3Texture( size.x, size.y, data );
}
#endif

// static
GLTextureRectangle* GLTextureRectangle::createUnsignedByte4Texture( int width, int height, const uint8_t* data )
{
	GLTexture::GLTextureInternalFormat internalFormat = GLTexture::RGBA_8_BYTE_UNORM;

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );

	return pTexture;
}

// static
GLTextureRectangle* GLTextureRectangle::createUnsignedByte4Texture( const Vector2i& size, const uint8_t* data )
{
	return GLTextureRectangle::createUnsignedByte4Texture( size.x, size.y, data );
}

// static
GLTextureRectangle* GLTextureRectangle::createFloat1Texture( int width, int height, int nBitsPerComponent, const float* data )
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

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_RED, GL_FLOAT, data );

	return pTexture;
}

// static
GLTextureRectangle* GLTextureRectangle::createFloat1Texture( const Vector2i& size, int nBitsPerComponent, const float* data )
{
	return createFloat1Texture( size.x, size.y, nBitsPerComponent, data );
}

// static
GLTextureRectangle* GLTextureRectangle::createFloat2Texture( int width, int height, int nBitsPerComponent, const float* data )
{
	GLTexture::GLTextureInternalFormat internalFormat;

	switch( nBitsPerComponent )
	{
	case 16:
    	internalFormat = GLTexture::RG_16_FLOAT;
		break;
	case 32:
		internalFormat = GLTexture::RG_32_FLOAT;
		break;
	default:
		fprintf( stderr, "Floating point texture nBits must be 16 or 32 bits!\n" );
		assert( false );
	}

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_RG, GL_FLOAT, data );

	return pTexture;
}

// static
GLTextureRectangle* GLTextureRectangle::createFloat3Texture( int width, int height, int nBits, const float* data )
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

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_RGB, GL_FLOAT, data );

	return pTexture;
}

// static
GLTextureRectangle* GLTextureRectangle::createFloat4Texture( int width, int height, int nBits, const float* data )
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

	GLTextureRectangle* pTexture = GLTextureRectangle::createTextureRectangle( width, height, internalFormat );

	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, internalFormat, width, height, 0, GL_RGBA, GL_FLOAT, data );

	return pTexture;
}

// static
GLTextureRectangle* GLTextureRectangle::createFloat4Texture( const Vector2i& size, int nBitsPerComponent, const float* data )
{
	return GLTextureRectangle::createFloat4Texture( size.x, size.y, nBitsPerComponent, data );
}

void GLTextureRectangle::setData( shared_ptr< Image1f > image )
{
	return setFloat1Data( image->pixels() );
}

#if 0
void GLTextureRectangle::setData( shared_ptr< Image3ub > image )
{
	return setUnsignedByte3Data( image->pixels() );
}
#endif

void GLTextureRectangle::setData( shared_ptr< Image4ub > image )
{
	return setUnsignedByte4Data( image->pixels() );
}

void GLTextureRectangle::setData( shared_ptr< Image4f > image )
{
	return setFloat4Data( image->pixels() );
}

#if 0
shared_ptr< Image3ub > GLTextureRectangle::getImage3ub( shared_ptr< Image3ub > output )
{
	if( output.get() == NULL )
	{
		output = new Image3ub( size() );
	}

	// TODO: level, etc
	getUnsignedByte3Data( output->pixels() );

	return output;
}
#endif

shared_ptr< Image1f > GLTextureRectangle::getImage1f( shared_ptr< Image1f > output )
{
	if( !output )
	{
		output.reset( new Image1f( size() ) );
	}

	// TODO: level, etc
	getFloat1Data( output->pixels() );

	return output;
}

shared_ptr< Image4f > GLTextureRectangle::getImage4f( shared_ptr< Image4f > output )
{
	if( !output )
	{
		output.reset( new Image4f( size() ) );
	}

	// TODO: level, etc
	getFloat4Data( output->pixels() );

	return output;
}

void GLTextureRectangle::setFloat1Data( const float* data )
{
	bind();
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, getInternalFormat(), m_width, m_height, 0, GL_LUMINANCE, GL_FLOAT, data );
}

void GLTextureRectangle::setFloat3Data( const float* data )
{
	bind();
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, getInternalFormat(), m_width, m_height, 0, GL_RGBA, GL_FLOAT, data );
}

void GLTextureRectangle::setFloat4Data( const float* data )
{
	bind();
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, getInternalFormat(), m_width, m_height, 0, GL_RGBA, GL_FLOAT, data );
}

void GLTextureRectangle::setUnsignedByte1Data( const uint8_t* data )
{
	bind();
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, getInternalFormat(), m_width, m_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data );
}

void GLTextureRectangle::setUnsignedByte3Data( const uint8_t* data )
{
	bind();
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, getInternalFormat(), m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
}

void GLTextureRectangle::setUnsignedByte4Data( const uint8_t* data )
{
	bind();
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, getInternalFormat(), m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
}

int GLTextureRectangle::numElements()
{
	return m_width * m_height;
}

int GLTextureRectangle::getWidth()
{
	return m_width;
}

int GLTextureRectangle::getHeight()
{
	return m_height;
}

Vector2i GLTextureRectangle::size()
{
	return Vector2i( m_width, m_height );
}

void GLTextureRectangle::setAllWrapModes( GLint iMode )
{
	bind();
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, iMode );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, iMode );
}

// virtual
void GLTextureRectangle::dumpToCSV( QString filename )
{
	// TODO: use getData(), use it to get the right formats

    vector< float > pixels( 4 * m_width * m_height );
	getFloat4Data( pixels.data() );

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
}

// virtual
void GLTextureRectangle::dumpToTXT( QString filename, GLint level, GLenum format, GLenum type )
{
	assert( level == 0 );
	assert( format == GL_RGBA );
	assert( type == GL_FLOAT );

	bind();

    vector< float > pixels( 4 * m_width * m_height );
	// TODO: use getData(), use it to get the right formats
	glGetTexImage( GL_TEXTURE_RECTANGLE, level, format, type, pixels.data() );

	FILE* fp = fopen( qPrintable( filename ), "w" );
	fprintf( fp, "width = %d, height = %d\n", m_width, m_height );
	fprintf( fp, "{pixel number} [float number] (rasterX, rasterY), ((openglX, openglY)) : <r, g, b, a>\n" );
	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			int k = 4 * ( y * m_width + x );

			fprintf( fp, "{%d} [%d] (%d, %d) ((%d,%d)): <%f, %f, %f, %f>\n", k / 4, k, x, m_height - y - 1, x, y, pixels[k], pixels[k+1], pixels[k+2], pixels[k+3] );
			k += 4;
		}
	}

	fclose( fp );
}

void GLTextureRectangle::savePPM( QString filename )
{
	bind();
	// TODO: use getData(), use it to get the right formats
    // TODO: bindless graphics

    vector< uint8_t > pixels( 3 * m_width * m_height );
	getUnsignedByte3Data( pixels.data() );
    Array2DView< ubyte3 > view( pixels.data(), m_width, m_height );
    PortablePixelMapIO::writeRGB( filename, view );
}

void GLTextureRectangle::savePNG( QString filename )
{
    vector< uint8_t > pixels( 4 * m_width * m_height );
	getUnsignedByte4Data( pixels.data() );

    // TODO: use a view
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

void GLTextureRectangle::savePFM( QString filename )
{
	vector< float > pixels( 3 * m_width * m_height );
	getFloat3Data( pixels.data() );
    Array2DView< Vector3f > view( pixels.data(), m_width, m_height );
	PortableFloatMapIO::writeRGB( filename, view );
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
GLTextureRectangle* GLTextureRectangle::createTextureRectangle( int width, int height,
										  GLTexture::GLTextureInternalFormat internalFormat )
{
	GLTextureRectangle* pTexture = new GLTextureRectangle( width, height, internalFormat );
	pTexture->setAllWrapModes( GL_CLAMP );
	pTexture->setFilterMode( GLTexture::GLTextureFilterMode_NEAREST, GLTexture::GLTextureFilterMode_NEAREST );
	return pTexture;
}

GLTextureRectangle::GLTextureRectangle( int width, int height,
						 GLTexture::GLTextureInternalFormat internalFormat ) :

    GLTexture( GL_TEXTURE_RECTANGLE, internalFormat ),
    m_width( width ),
    m_height( height )

{
	assert( width > 0 );
	assert( height > 0 );
}
