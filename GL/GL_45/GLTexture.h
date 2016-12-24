#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/BasicTypes.h>
#include "libcgt/GL/GLImageInternalFormat.h"
#include "libcgt/GL/GLImageFormat.h"

class Vector2f;
class Vector3f;
class Vector4f;

// TODO: consider folding sampler into this.
//   or make a new hierarchy for GLBindlessTexture.
// TODO: move mipmap levels and clear() here.
// TODO: fix GLTexture1D and GLTexture3D.
// TOD: migrate cube maps to ARB_DSA. Needs documentation.
class GLTexture
{
public:

    enum class Target : GLenum
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

    enum class SwizzleSource : GLenum
    {
        RED = GL_TEXTURE_SWIZZLE_R,
        GREEN = GL_TEXTURE_SWIZZLE_G,
        BLUE = GL_TEXTURE_SWIZZLE_B,
        ALPHA = GL_TEXTURE_SWIZZLE_A
        // Set RGBA using setSwizzleRGBA().
    };

    enum class SwizzleTarget : GLint
    {
        RED = GL_RED,
        GREEN = GL_GREEN,
        BLUE = GL_BLUE,
        ALPHA = GL_ALPHA,
        ZERO = GL_ZERO,
        ONE = GL_ONE
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

    // TODO: other source formats and types.
    void clear( const uint8x4& clearValue, int level );
    void clear( float clearValue, GLImageFormat srcFormat, int level );
    void clear( const Vector2f& clearValue, int level );
    void clear( const Vector3f& clearValue, int level );
    void clear( const Vector4f& clearValue, int level );

    // TODO(multi_bind):
    // glBindTextures( GLuint firstTextureUnitIndex, int count, GLuint* textureIds )
    // glBindSamplers()
    // glBindTexturesSamplers()

    GLuint id() const;
    Target target() const;
    GLenum glTarget() const;
    GLImageInternalFormat internalFormat() const;
    GLenum glInternalFormat() const;
    GLsizei numMipMapLevels() const;

    // TODO(jiawen): get swizzle?

    // Set the texture swizzle, such that a shader or fixed-function fragment
    // operation reading from source produces the target instead.
    void setSwizzle( SwizzleSource source, SwizzleTarget target );

    // Sets all 4 source channels simultaneously.
    void setSwizzleRGBA( SwizzleTarget targets[ 4 ] );

    void generateMipMaps();

protected:

    GLTexture( Target target, GLImageInternalFormat internalFormat,
              GLsizei nMipMapLevels );
    GLTexture( GLTexture&& move );
    GLTexture& operator = ( GLTexture&& move );

private:

    GLuint m_id = 0;
    Target m_target;
    GLImageInternalFormat m_internalFormat;
    GLsizei m_nMipMapLevels = 0;

    void destroy();
};
