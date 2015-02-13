#pragma once

#include <GL/glew.h>

#include <vecmath/Matrix4f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Rect2f.h>
#include <vecmath/Box3f.h>

class GLUtilities
{
public:

	// Clears the standard clamped to [0,1] depth buffer, regardless of representation.
	static void clearDepth( GLclampd d = 0 );

	// Clears the unclamped floating point depth buffer to d.
	static void clearDepthNV( double d );

	static Matrix4f getProjectionMatrix();
	static Matrix4f getModelviewMatrix();

	// same thing as getProjectionMatrix() * getModelviewMatrix()
	static Matrix4f getModelviewProjectionMatrix();
	
	static void setupOrthoCamera( int viewportWidth, int viewportHeight );

	static void setupOrthoCamera( const Vector2i& viewportSize );

    // Gets the 2D rectangular viewport, as { 0, 0, width, height }.
	static Rect2f getViewport();
    // Gets the 3D box viewport, which is the 2D viewport
    // and depth range: as { 0, 0, zNear, width, height, zFar - zNear }.
	static Box3f getViewport3D();

    // Assumes that vp is a standard rectangle.
	static void setViewport( const Rect2f& vp );

	// Specify mapping of depth values from NDC [-1,1] to window coordinates [zNear, zFar]
	// ARB_depth_buffer_float is still clamped.
	static void setDepthRange( GLclampd zNear, GLclampd zFar );
	// NV_depth_buffer_float is unclamped and takes doubles
	// (The default is still [0,1])
	static void setDepthRangeNV( double zNear, double zFar );

	static void drawCross( float width, float height );

	static float* readDepthBuffer( int x, int y, int width, int height );
	static void dumpDepthBufferText( int x, int y, int width, int height, const char* szFilename );
	static void dumpFrameBufferLuminanceText( int x, int y, int width, int height, const char* szFilename );
	static void dumpFrameBufferLuminanceBinary( int x, int y, int width, int height, const char* szFilename );
	static void dumpFrameBufferRGBABinary( int x, int y, int width, int height, const char* szFilename );	
	static void dumpFrameBufferRGBAText( int x, int y, int width, int height, const char* szFilename );	

	static void printGLRenderer();
	static void printGLVendor();
	static void printGLVersion();	
	
	// Prints the last GL error.
	// Returns true if there was *no error*.
	static bool printLastError();

private:

};
