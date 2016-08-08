#include <cstdio>

#if defined _WIN32 || defined _WIN64
#include <windows.h>
#endif

#include "common/ArrayUtils.h"
#include "io/PNGIO.h"
#include "io/PortableFloatMapIO.h"

#ifdef GL_PLATFORM_ES_31
#include "../GLES_31/GLTexture2D.h"
#endif
#ifdef GL_PLATFORM_45
#include "../GL_45/GLTexture2D.h"
#endif
#include "GLUtilities.h"

using libcgt::core::arrayutils::componentView;

// static
void GLUtilities::clearDepth( GLfloat d )
{
    glClearDepthf( d );
}

#ifdef GL_PLATFORM_45
// static
void GLUtilities::clearDepth( GLdouble d )
{
    glClearDepth( d );
}

// static
void GLUtilities::clearDepthNV( GLdouble d )
{
    glClearDepthdNV( d );
}
#endif

// static
Rect2i GLUtilities::getViewport()
{
    int vp[4];
    glGetIntegerv( GL_VIEWPORT, vp );
    return{ { vp[0], vp[1] }, { vp[2], vp[3] } };
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
    return{ { vp[0], vp[1], zNear }, { vp[2], vp[3], zFar - zNear } };
}

// static
void GLUtilities::setViewport( const Rect2i& vp )
{
    glViewport( vp.origin.x, vp.origin.y, vp.size.x, vp.size.y );
}

// static
void GLUtilities::setDepthRange( GLfloat zNear, GLfloat zFar )
{
    glDepthRangef( zNear, zFar );
}

#ifdef GL_PLATFORM_45
// static
void GLUtilities::setDepthRange( GLdouble zNear, GLdouble zFar )
{
    glDepthRange( zNear, zFar );
}

// static
void GLUtilities::setDepthRangeNV( double zNear, double zFar )
{
    glDepthRangedNV( zNear, zFar );
}
#endif

#ifdef GL_PLATFORM_45
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
void GLUtilities::saveTextureToFile( GLTexture2D* pTexture, const std::string& filename )
{
    if( pTexture->internalFormat() == GLImageInternalFormat::RGBA8 )
    {
        Array2D< uint8x4 > rgba8( pTexture->size() );
        pTexture->get( rgba8 );
        PNGIO::write( filename, rgba8 );
    }
    else if( pTexture->internalFormat() == GLImageInternalFormat::R32F )
    {
        Array2D< float > r32f( pTexture->size() );
        pTexture->get( r32f );
        PortableFloatMapIO::write( filename, r32f );
    }
    else if( pTexture->internalFormat() == GLImageInternalFormat::RGBA32F )
    {
        Array2D< Vector4f > rgba32f( pTexture->size() );
        pTexture->get( rgba32f );
        Array2DView< Vector3f > rgb32f =
            componentView< Vector3f >( rgba32f.writeView(), 0 );
        PortableFloatMapIO::write( filename, rgb32f );
    }
}
#endif

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
void GLUtilities::printLastError()
{
    fprintf( stderr, "%s\n", getLastErrorString().c_str() );
}

// static
std::string GLUtilities::getLastErrorString()
{
    GLenum glError = glGetError();
    switch( glError )
    {
    case GL_NO_ERROR:
        return "GL_NO_ERROR";
    case GL_INVALID_ENUM:
        return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
        return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
        return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY:
        return "GL_OUT_OF_MEMORY";
#ifdef GL_PLATFORM_45
    case GL_STACK_UNDERFLOW:
        return "GL_STACK_UNDERFLOW";
    case GL_STACK_OVERFLOW:
        return "GL_STACK_OVERFLOW";
#endif
    default:
        // TODO: to_string hex
        return "Unknown error: " +
            std::to_string( static_cast< unsigned int >( glError ) );
    }
}
