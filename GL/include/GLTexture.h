#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/BasicTypes.h>
#include "GLImageInternalFormat.h"
#include "GLImageFormat.h"

class GLTexture
{
public:

	// Returns the current active texture unit.
	static GLenum activeTextureUnit();

	// Returns the maximum number of texture image units
	// that can be bound per pipeline stage.
	static int maxTextureImageUnits();

	// Returns the maximum number of texture image units
	// across the entire pipeline.
	static int maxCombinedTextureImageUnits();
	
	// Max width and height
	static int maxSize1D2D();

	// Max width, height, and depth
	static int maxSize3D();

	// Max for any face
	static int maxSizeCubeMap();

	virtual ~GLTexture();

	// Binds this texture object to the texture unit;
	void bind( GLenum texunit = GL_TEXTURE0 );

	// Unbinds this texture from the texture unit.
	void unbind( GLenum texunit = GL_TEXTURE0 );

	GLuint id();
	GLenum target(); // TODO: make target also enum class
	GLImageInternalFormat internalFormat();

	virtual void getFloat1Data( float* afOutput, int level = 0 );
	virtual void getFloat3Data( float* afOutput, int level = 0 );
	virtual void getFloat4Data( float* afOutput, int level = 0 );

	virtual void getUnsignedByte1Data( uint8_t* aubOutput, int level = 0 );
	virtual void getUnsignedByte3Data( uint8_t* aubOutput, int level = 0 );
	virtual void getUnsignedByte4Data( uint8_t* aubOutput, int level = 0 );

protected:
	
	GLTexture( GLenum target, GLImageInternalFormat internalFormat );

private:

	void getTexImage( GLint level, GLenum format, GLenum type, void* avOutput );

	GLenum m_target;
	GLuint m_id;	
	GLImageInternalFormat m_internalFormat;
};
