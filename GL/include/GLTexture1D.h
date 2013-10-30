#ifndef GL_TEXTURE_1D_H
#define GL_TEXTURE_1D_H

#include <cstdint>

#include <GL/glew.h>

#include <common/BasicTypes.h>

#include "GLTexture.h"

class GLTexture1D : public GLTexture
{
public:

	static void setEnabled( bool bEnabled );

	// creates a ubyte1 (8 bits luminance for each pixel) texture
	static GLTexture1D* createUnsignedByte1Texture( int width, const uint8_t* data = 0 );

	// creates a ubyte3 (8 bits for each component) texture
	static GLTexture1D* createUnsignedByte3Texture( int width, const uint8_t* data = 0 );

	// creates a ubyte4 (8 bits for each component) texture
	static GLTexture1D* createUnsignedByte4Texture( int width, const uint8_t* data = 0 );

	// creates a float1 texture
	static GLTexture1D* createFloat1Texture( int width, int nBitsPerComponent = 32, const float* data = 0 );

	// creates a float3 texture
	static GLTexture1D* createFloat3Texture( int width, int nBitsPerComponent = 32, const float* data = 0 );

	// creates a float4 texture	
	static GLTexture1D* createFloat4Texture( int width, int nBitsPerComponent = 32, const float* data = 0 );

	// uploads float array to hardware
	// by default, assumes that the entire texture is being updated
	// (pass in width and height = 0)
	// otherwise, specify the rectangle
	void setFloat1Data( const float* data, int xOffset = 0, int width = 0 );
	void setFloat3Data( const float* data, int xOffset = 0, int width = 0 );
	void setFloat4Data( const float* data, int xOffset = 0, int width = 0 );

	// uploads unsigned byte array to hardware
	// by default, assumes that the entire texture is being updated
	// (pass in width and height = 0)
	// otherwise, specify the rectangle
	void setUnsignedByte1Data( const uint8_t* data,
		int xOffset = 0, int width = 0 );
	void setUnsignedByte3Data( const uint8_t* data,
		int xOffset = 0, int width = 0 );
	void setUnsignedByte4Data( const uint8_t* data,
		int xOffset = 0, int width = 0 );

	int getWidth();

	// sets the wrap mode of the currently bound texture
	virtual void setAllWrapModes( GLint iMode );

	virtual uint8_t* getUByteData( GLint level = 0 );
	virtual float* getFloatData( GLint level = 0 );

	virtual void dumpToCSV( QString filename );
	virtual void dumpToTXT( QString filename, GLint level = 0, GLenum format = GL_RGBA, GLenum type = GL_FLOAT );
	virtual void dumpToPPM( QString filename, GLint level = 0, GLenum format = GL_RGBA, GLenum type = GL_FLOAT );

private:

	// wrapper around constructor
	// creates the texture, sets wrap mode to clamp, and filter mode to nearest
	static GLTexture1D* createTexture1D( int width,	GLTexture::GLTextureInternalFormat internalFormat );

	GLTexture1D( int width,	GLTexture::GLTextureInternalFormat internalFormat );

	int m_width;
};

#endif
