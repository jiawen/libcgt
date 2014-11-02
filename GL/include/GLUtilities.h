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

	static Rect2f getViewport();
	static Box3f getViewport3D(); // including depth range, usually [0,1]

	static void setViewport( const Rect2f& vp );

	// Specify mapping of depth values from NDC [-1,1] to window coordinates [zNear, zFar]
	// ARB_depth_buffer_float is still clamped
	static void setDepthRange( GLclampd zNear, GLclampd zFar );
	// NV_depth_buffer_float is unclamped and takes doubles
	// (The default is still [0,1])
	static void setDepthRangeNV( double zNear, double zFar );

	// draw a screen aligned quad.  if zeroOneTextureCoordinates are true,
	// then the tex coordinates range from 0 to 1.  Otherwise, they will range
	// from 0 to width and 0 to height
	static void drawQuad( int width, int height, bool zeroOneTextureCoordinates = false, bool tTextureCoordinateZeroOnBottom = true );
	static void drawQuad( const Vector2i& size, bool zeroOneTextureCoordinates = false, bool tTextureCoordinateZeroOnBottom = true );
	static void drawQuad( int x, int y, int width, int height, bool zeroOneTextureCoordinates = false, bool tTextureCoordinateZeroOnBottom = true );

	static void drawQuad( const Rect2f& position, const Rect2f& textureCoordinates, bool flipTextureCoordinatesUpDown = false );

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
