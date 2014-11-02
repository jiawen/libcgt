#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/BasicTypes.h>

#include "GLTexture.h"

class GLTexture1D : public GLTexture
{
public:

	GLTexture1D( int width, GLImageInternalFormat internalFormat );

    // TODO: use set on a view, like GLTexture2D.
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

	int width() const;

private:	

	int m_width;
};
