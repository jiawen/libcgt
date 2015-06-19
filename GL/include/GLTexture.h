#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/BasicTypes.h>
#include "GLImageInternalFormat.h"
#include "GLImageFormat.h"

// TODO: move mipmap levels and clear() here.
// TODO: fix GLTexture1D and GLTexture3D.
// TOD: migrate cube maps to ARB_DSA. Needs documentation.
class GLTexture
{
public:

    enum class Target
    {
        // Standard targets.
        TEXTURE_1D = GL_TEXTURE_1D,
        TEXTURE_2D = GL_TEXTURE_2D,
        TEXTURE_3D = GL_TEXTURE_3D,

        // Special geometric targets.
        TEXTURE_CUBE_MAP = GL_TEXTURE_CUBE_MAP,
        TEXTURE_RECTANGLE = GL_TEXTURE_RECTANGLE, // basically deprecated

        // Array targets.
        TEXTURE_1D_ARRAY = GL_TEXTURE_1D_ARRAY,
        TEXTURE_2D_ARRAY = GL_TEXTURE_2D_ARRAY,
        TEXTURE_CUBE_MAP_ARRAY = GL_TEXTURE_CUBE_MAP_ARRAY,

        // Multisample targets.
        TEXTURE_2D_MULTISAMPLE = GL_TEXTURE_2D_MULTISAMPLE,
        TEXTURE_2D_MULTISAMPLE_ARRAY = GL_TEXTURE_2D_MULTISAMPLE_ARRAY,

        // Texture buffer.
        TEXTURE_BUFFER = GL_TEXTURE_BUFFER
    };

	// Returns the current active texture unit.
	static GLenum activeTextureUnit();

	// Returns the maximum number of texture image units
	// that can be bound per pipeline stage.
	static int maxTextureImageUnits();

	// Returns the maximum number of texture image units
	// across the entire pipeline.
	static int maxCombinedTextureImageUnits();

    // Max width.
	static int maxSize1D();

	// Max width and height.
	static int maxSize2D();

	// Max width, height, and depth.
	static int maxSize3D();

	// Max width/height for any individual face (must be square).
    // This corresponds to GL_MAX_ARRAY_TEXTURE_LAYERS.
	static int maxSizeCubeMap();

    // The maximum number of "layers" in an array texture (array length).
    static int maxArrayLayers();

	virtual ~GLTexture();

	// Binds this texture object to the texture unit;
	void bind( GLuint textureUnitIndex = 0 ) const;

	// Unbinds this texture from the texture unit.
	void unbind( GLuint textureUnitIndex = 0 ) const;

    // TODO(multi_bind):
    // glBindTextures( GLuint firstTextureUnitIndex, int count, GLuint* textureIds )
    // glBindSamplers()
    // glBindTexturesSamplers()

	GLuint id() const;
	GLenum target() const; // TODO: make target also enum class
	GLImageInternalFormat internalFormat() const;

protected:
	
	GLTexture( GLenum target, GLImageInternalFormat internalFormat );

private:

	GLenum m_target;
	GLuint m_id;	
	GLImageInternalFormat m_internalFormat;
};
