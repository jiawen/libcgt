#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/BasicTypes.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "GLTexture.h"

class GLTexture3D : public GLTexture
{
public:

	GLTexture3D( const Vector3i& size, GLImageInternalFormat internalFormat );

    int numElements() const;
	int width() const;
	int height() const;
	int depth() const;
	Vector3i size() const;

    // TODO: use views just like GLTexture2D(), call them set() and get()

	// uploads float array to hardware
	// by default, assumes that the entire texture is being updated
	// (pass in width and height = 0)
	// otherwise, specify the rectangle
	void setFloat1Data( const float* afData, int xOffset = 0, int yOffset = 0, int zOffset = 0, int width = 0, int height = 0, int depth = 0 );
	void setFloat3Data( const float* afData, int xOffset = 0, int yOffset = 0, int zOffset = 0, int width = 0, int height = 0, int depth = 0 );
	void setFloat4Data( const float* afData, int xOffset = 0, int yOffset = 0, int zOffset = 0, int width = 0, int height = 0, int depth = 0 );

	// uploads unsigned byte array to hardware
	// by default, assumes that the entire texture is being updated
	// (pass in width and height = 0)
	// otherwise, specify the rectangle
	void setUnsignedByte1Data( const uint8_t* data,
		int xOffset = 0, int yOffset = 0, int zOffset = 0,
		int width = 0, int height = 0, int depth = 0 );
	void setUnsignedByte3Data( const uint8_t* data,
		int xOffset = 0, int yOffset = 0, int zOffset = 0,
		int width = 0, int height = 0, int depth = 0 );
	void setUnsignedByte4Data( const uint8_t* data,
		int xOffset = 0, int yOffset = 0, int zOffset = 0,
		int width = 0, int height = 0, int depth = 0 );	

private:	

    Vector3i m_size;
};
