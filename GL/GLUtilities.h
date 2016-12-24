#pragma once

#include <string>

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

#include "libcgt/core/vecmath/Matrix4f.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Rect2i.h"
#include "libcgt/core/vecmath/Box3f.h"

class GLTexture2D;


// TODO: refactor some of these into a GLPIpelineState object.

class GLUtilities
{
public:

    // Clears the standard clamped to [0,1] depth buffer, regardless of representation.
    static void clearDepth( GLfloat d = 0 );

#ifdef GL_PLATFORM_45
    // Clears the standard clamped to [0,1] depth buffer, regardless of representation.
    static void clearDepth( GLdouble d = 0 );

    // Clears the unclamped floating point depth buffer to d.
    static void clearDepthNV( GLdouble d );
#endif

    // Gets the 2D rectangular viewport, as { 0, 0, width, height }.
    static Rect2i getViewport();

    // Gets the 3D box viewport, which is the 2D viewport
    // and depth range: as { 0, 0, zNear, width, height, zFar - zNear }.
    static Box3f getViewport3D();

    // Assumes that vp is a standard rectangle.
    static void setViewport( const Rect2i& vp );

    // Specify mapping of depth values from NDC [-1,1] to window coordinates [zNear, zFar]
    // ARB_depth_buffer_float is still clamped.
    static void setDepthRange( GLfloat zNear, GLfloat zFar );

#ifdef GL_PLATFORM_45
    // Specify mapping of depth values from NDC [-1,1] to window coordinates [zNear, zFar]
    // ARB_depth_buffer_float is still clamped.
    static void setDepthRange( GLdouble zNear, GLdouble zFar );

    // NV_depth_buffer_float is unclamped and takes doubles
    // (The default is still [0,1])
    static void setDepthRangeNV( double zNear, double zFar );
#endif

#ifdef GL_PLATFORM_45
    static float* readDepthBuffer( int x, int y, int width, int height );
    static void dumpDepthBufferText( int x, int y, int width, int height, const char* szFilename );
    static void dumpFrameBufferLuminanceText( int x, int y, int width, int height, const char* szFilename );
    static void dumpFrameBufferLuminanceBinary( int x, int y, int width, int height, const char* szFilename );
    static void dumpFrameBufferRGBABinary( int x, int y, int width, int height, const char* szFilename );
    static void dumpFrameBufferRGBAText( int x, int y, int width, int height, const char* szFilename );

    // Supported combinations of texture / image:
    // RGBA8 --> PNG
    // R32F --> PFM
    static void saveTextureToFile( GLTexture2D* pTexture, const std::string& filename );
#endif

    static void printGLRenderer();
    static void printGLVendor();
    static void printGLVersion();

    // Prints the last GL error.
    // Returns true if there was *no error*.
    static void printLastError();

    // Gets the last GL error and returns it as a string.
    static std::string getLastErrorString();

private:

};
