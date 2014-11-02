#include <cstdio>

#if defined _WIN32 || defined _WIN64
#include <windows.h>
#endif

#include "GLUtilities.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
void GLUtilities::clearDepthNV( double d )
{
	glClearDepth( d );

	glClearDepthdNV( d );
}

// static
Matrix4f GLUtilities::getProjectionMatrix()
{
	Matrix4f projectionMatrix;
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix );
	return projectionMatrix;
}

// static
Matrix4f GLUtilities::getModelviewMatrix()
{
	Matrix4f modelviewMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, modelviewMatrix );
	return modelviewMatrix;
}

// static
Matrix4f GLUtilities::getModelviewProjectionMatrix()
{
	return getProjectionMatrix() * getModelviewMatrix();
}

// static
void GLUtilities::setupOrthoCamera( int viewportWidth, int viewportHeight )
{
	glViewport( 0, 0, viewportWidth, viewportHeight );

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluOrtho2D( 0, viewportWidth, 0, viewportHeight );

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
}

// static
void GLUtilities::setupOrthoCamera( const Vector2i& viewportSize )
{
	setupOrthoCamera( viewportSize.x, viewportSize.y );
}

// static
Rect2f GLUtilities::getViewport()
{
	float vp[4];
	glGetFloatv( GL_VIEWPORT, vp );
	return Rect2f( vp[0], vp[1], vp[2], vp[3] );
}

// static
Box3f GLUtilities::getViewport3D()
{
	float vp[4];
	glGetFloatv( GL_VIEWPORT, vp );

	float dr[2];
	glGetFloatv( GL_DEPTH_RANGE, dr );

	float zNear = dr[0];
	float zFar = dr[1];
	return Box3f( vp[0], vp[1], zNear, vp[2], vp[3], zFar - zNear );
}

// static
void GLUtilities::setViewport( const Rect2f& vp )
{
	glViewport( vp.left(), vp.bottom(), vp.width(), vp.height() );
}

// static
void GLUtilities::setDepthRange( GLclampd zNear, GLclampd zFar )
{
	glDepthRange( zNear, zFar );
}

// static
void GLUtilities::setDepthRangeNV( double zNear, double zFar )
{
	glDepthRangedNV( zNear, zFar );
}

// static
void GLUtilities::drawQuad( int width, int height, bool zeroOneTextureCoordinates, bool tTextureCoordinateOnBottom )
{
	drawQuad( 0, 0, width, height, zeroOneTextureCoordinates, tTextureCoordinateOnBottom );
}

// static
void GLUtilities::drawQuad( const Vector2i& size, bool zeroOneTextureCoordinates, bool tTextureCoordinateZeroOnBottom )
{
	drawQuad( size.x, size.y, zeroOneTextureCoordinates, tTextureCoordinateZeroOnBottom );
}

// static
void GLUtilities::drawQuad( int x, int y, int width, int height, bool zeroOneTextureCoordinates, bool tTextureCoordinateOnBottom )
{
	float texCoords[8];
	float positions[8];

	if( zeroOneTextureCoordinates )
	{
		if( tTextureCoordinateOnBottom )
		{
			texCoords[0] = 0;
			texCoords[1] = 0;

			texCoords[2] = 1;
			texCoords[3] = 0;
			
			texCoords[4] = 1;
			texCoords[5] = 1;
			
			texCoords[6] = 0;
			texCoords[7] = 1;
		}
		else
		{
			texCoords[0] = 0;
			texCoords[1] = 1;

			texCoords[2] = 1;
			texCoords[3] = 1;

			texCoords[4] = 1;
			texCoords[5] = 0;

			texCoords[6] = 0;
			texCoords[7] = 0;
		}
	}
	else
	{
		if( tTextureCoordinateOnBottom )
		{
			texCoords[0] = 0;
			texCoords[1] = 0;

			texCoords[2] = static_cast< float >( width );
			texCoords[3] = 0;

			texCoords[4] = static_cast< float >( width );
			texCoords[5] = static_cast< float >( height );

			texCoords[6] = 0;
			texCoords[7] = static_cast< float >( height );
		}
		else
		{
			texCoords[0] = 0;
			texCoords[1] = static_cast< float >( height );

			texCoords[2] = static_cast< float >( width );
			texCoords[3] = static_cast< float >( height );

			texCoords[4] = static_cast< float >( width );
			texCoords[5] = 0;

			texCoords[6] = 0;
			texCoords[7] = 0;
		}
	}

	positions[0] = static_cast< float >( x );
	positions[1] = static_cast< float >( y );

	positions[2] = static_cast< float >( x + width );
	positions[3] = static_cast< float >( y );

	positions[4] = static_cast< float >( x + width );
	positions[5] = static_cast< float >( y + height );

	positions[6] = static_cast< float >( x );
	positions[7] = static_cast< float >( y + height );

	glBegin( GL_QUADS );

		for( int i = 0; i < 8; i += 2 )
		{
			glTexCoord2f( texCoords[i], texCoords[ i + 1 ] );
			glVertex2f( positions[i], positions[ i + 1 ] );
		}

	glEnd();
}

// static
void GLUtilities::drawQuad( const Rect2f& position, const Rect2f& textureCoordinates, bool flipTextureCoordinatesUpDown )
{
	glBegin( GL_QUADS );

	if( flipTextureCoordinatesUpDown )
	{
		glTexCoord2fv( textureCoordinates.topLeft() );
		glVertex2fv( position.bottomLeft() );
		
		glTexCoord2fv( textureCoordinates.topRight() );
		glVertex2fv( position.bottomRight() );

		glTexCoord2fv( textureCoordinates.bottomRight() );
		glVertex2fv( position.topRight() );

		glTexCoord2fv( textureCoordinates.bottomLeft() );
		glVertex2fv( position.topLeft() );
	}
	else
	{
		glTexCoord2fv( textureCoordinates.bottomLeft() );
		glVertex2fv( position.bottomLeft() );

		glTexCoord2fv( textureCoordinates.bottomRight() );
		glVertex2fv( position.bottomRight() );
		
		glTexCoord2fv( textureCoordinates.topRight() );
		glVertex2fv( position.topRight() );
		
		glTexCoord2fv( textureCoordinates.topLeft() );
		glVertex2fv( position.topLeft() );
	}

	glEnd();
}

// static
void GLUtilities::drawCross( float width, float height )
{	
	glBegin( GL_LINES );

	glVertex2f( 0.f, 0.f );
	glVertex2f( width, height );

	glVertex2f( width, 0.f );
	glVertex2f( 0.f, height );

	glEnd();
}

// static
float* GLUtilities::readDepthBuffer( int x, int y, int width, int height )
{
	float* depthBuffer = new float[ width * height ];
	glReadPixels( x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depthBuffer );
	return depthBuffer;
}

// static
void GLUtilities::dumpDepthBufferText( int x, int y, int width, int height, const char* szFilename )
{
	float* depthBuffer = GLUtilities::readDepthBuffer( x, y, width, height );

	FILE* fp = fopen( szFilename, "w" );
	fprintf( fp, "Width = %d, Height = %d\n", width, height );

	int k = 0;
	for( int y = 0; y < height; ++y )
	{
		for( int x = 0; x < width; ++x )
		{
			fprintf( fp, "(%d, %d): %f\n", x, y, depthBuffer[k] );
			++k;
		}
	}
	fclose( fp );

	delete[] depthBuffer;
}

// static
void GLUtilities::dumpFrameBufferLuminanceBinary( int x, int y, int width, int height, const char* szFilename )
{
	int dimensions[2];
	dimensions[0] = width;
	dimensions[1] = height;

	float* pixels = new float[ width * height ];
	glReadPixels( x, y, width, height, GL_RED, GL_FLOAT, pixels );
	
	FILE* fp = fopen( szFilename, "wb" );
	fwrite( dimensions, sizeof( int ), 2, fp );
	fwrite( pixels, sizeof( float ), width * height, fp );
	fclose( fp );

	delete[] pixels;
}

// static
void GLUtilities::dumpFrameBufferLuminanceText( int x, int y, int width, int height, const char* szFilename )
{
	float* pixels = new float[ width * height ];
    glReadPixels( x, y, width, height, GL_RED, GL_FLOAT, pixels );

	FILE* fp = fopen( szFilename, "w" );
	fprintf( fp, "Width = %d, Height = %d\n", width, height );
	
	int k = 0;
	for( int y = 0; y < height; ++y )
	{
		for( int x = 0; x < width; ++x )
		{
			fprintf( fp, "(%d, %d): %f\n", x, y, pixels[k] );
			++k;
		}
	}
	fclose( fp );

	delete[] pixels;
}

// static
void GLUtilities::dumpFrameBufferRGBABinary( int x, int y, int width, int height, const char* szFilename )
{
	int dimensions[2];
	dimensions[0] = width;
	dimensions[1] = height;

	float* pixels = new float[ 4 * width * height ];
	glReadPixels( x, y, width, height, GL_RGBA, GL_FLOAT, pixels );

	FILE* fp = fopen( szFilename, "wb" );
	fwrite( dimensions, sizeof( int ), 2, fp );
	fwrite( pixels, sizeof( float ), 4 * width * height, fp );
	fclose( fp );

	delete[] pixels;
}

// static
void GLUtilities::dumpFrameBufferRGBAText( int x, int y, int width, int height, const char* szFilename )
{
	float* pixels = new float[ 4 * width * height ];
	glReadPixels( x, y, width, height, GL_RGBA, GL_FLOAT, pixels );

	int k = 0;
	FILE* fp = fopen( szFilename, "w" );

	for( int yy = 0; yy < height; ++yy )
	{
		for( int xx = 0; xx < width; ++xx )
		{
			fprintf( fp, "(%d, %d): (%f, %f, %f, %f)\n", xx + x, yy + y, pixels[k], pixels[k+1], pixels[k+2], pixels[k+3] );
			k += 4;
		}
	}

	fclose( fp );

	delete[] pixels;
}

// static
void GLUtilities::printGLRenderer()
{
	printf( "%s\n", glGetString( GL_RENDERER ) );
}

// static
void GLUtilities::printGLVendor()
{
	printf( "%s\n", glGetString( GL_VENDOR ) );
}

// static
void GLUtilities::printGLVersion()
{
	printf( "%s\n", glGetString( GL_VERSION ) );
}

// static
bool GLUtilities::printLastError()
{
	bool noErrorOccurred = false;

	GLenum glError = glGetError();
	switch( glError )
	{
	case GL_NO_ERROR:
		fprintf( stderr, "GL_NO_ERROR\n" );
		noErrorOccurred = true;
		break;
	case GL_INVALID_ENUM:
		fprintf( stderr, "GL_INVALID_ENUM\n" );
		break;
	case GL_INVALID_VALUE:
		fprintf( stderr, "GL_INVALID_VALUE\n" );
		break;
	case GL_INVALID_OPERATION:
		fprintf( stderr, "GL_INVALID_OPERATION\n" );
		break;
	case GL_STACK_OVERFLOW:
		fprintf( stderr, "GL_STACK_OVERFLOW\n" );
		break;
	case GL_STACK_UNDERFLOW:
		fprintf( stderr, "GL_STACK_UNDERFLOW\n" );
		break;
	case GL_OUT_OF_MEMORY:
		fprintf( stderr, "GL_OUT_OF_MEMORY\n" );
		break;
	default:
		fprintf( stderr, "Unknown error: 0x%8x!\n", static_cast< unsigned int >( glError ) );
		
		// HACK:
		// apparently, when you bind a framebuffer object
		// it sets the error state to GL_FRAMEBUFFER_INCOMPLETE
		// even though I haven't attached a texture yet!
		noErrorOccurred = true;
	}

	return noErrorOccurred;
}
