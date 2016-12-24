#pragma once

#include <cstdint>

#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

#include <common/BasicTypes.h>
#include "GLImageInternalFormat.h"
#include "GLImageFormat.h"

class GLTexture
{
public:

    enum class Target
    {
        // Standard targets.
        TEXTURE_2D = GL_TEXTURE_2D,
        TEXTURE_3D = GL_TEXTURE_3D,

        // Special geometric targets.
        TEXTURE_CUBE_MAP = GL_TEXTURE_CUBE_MAP,

        // Array targets.
        TEXTURE_2D_ARRAY = GL_TEXTURE_2D_ARRAY,

        // Multisample targets.
        TEXTURE_2D_MULTISAMPLE = GL_TEXTURE_2D_MULTISAMPLE,

        // External target.
        TEXTURE_EXTERNAL_OES = GL_TEXTURE_EXTERNAL_OES
    };

    // Returns the current active texture unit.
    static GLenum activeTextureUnit();

    // Returns the maximum number of texture image units
    // that can be bound per pipeline stage.
    static int maxTextureImageUnits();

    // Returns the maximum number of texture image units
    // across the entire pipeline.
    static int maxCombinedTextureImageUnits();

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

    // Binds this texture object to the currently active texture unit.
    // The target is determined by the texture subclass.
    void bind() const;

    // Unbinds the texture in the current active texture unit.
    void unbind() const;

    GLuint id() const;
    Target target() const;
    GLenum glTarget() const;
    GLImageInternalFormat internalFormat() const;
    GLenum glInternalFormat() const;
    GLsizei numMipMapLevels() const;

    // If this texture is bound to a target, automatically generates mipmaps
    // from the base image.
    void generateMipMaps();

protected:

    GLTexture( Target target, GLImageInternalFormat internalFormat,
        GLsizei nMipMapLevels );

private:

    GLuint m_id;
    Target m_target;
    GLImageInternalFormat m_internalFormat;
    GLsizei m_nMipMapLevels;
};

