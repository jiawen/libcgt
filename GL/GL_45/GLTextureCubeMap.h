#pragma once

#include <GL/glew.h>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/vecmath/Rect2f.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

#include "GLTexture.h"

class GLSamplerObject;

enum class GLCubeMapFace
{
    POSITIVE_X = GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    NEGATIVE_X = GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    POSITIVE_Y = GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    NEGATIVE_Y = GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    POSITIVE_Z = GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    NEGATIVE_Z = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
};

// When attached to a layered framebuffer, layers are numbered 0 through 5 and
// correspond to +x, -x, +y, -y, +z, -z.
// If it's a mipmapped cube map:
// array_layer = floor(layer / 6).
// face_index = layer % 6.
// I.e., faces are packed all together as:
// [ layer0( f0 ... f5 ), layer1( f0 ... f5 ), ... ]
class GLTextureCubeMap : public GLTexture
{
public:

    // TODO: enable global seamless texture filtering:
    // glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS).

    GLTextureCubeMap( int sideLength, GLImageInternalFormat internalFormat );

    int sideLength() const;

    // TODO: clear with a rectangle
    void clear( const uint8x4& clearValue = uint8x4{ 0, 0, 0, 0 } );
    void clear( float clearValue = 0.f, GLImageFormat format = GLImageFormat::RED );
    void clear( const Vector4f& clearValue = Vector4f( 0, 0, 0, 0 ) );

    // Data must be packed().
    // Only accepts RGBA and BGRA for now.
    bool set( GLCubeMapFace face, Array2DReadView< uint8x4 > data,
        GLImageFormat format = GLImageFormat::RGBA,
        const Vector2i& dstOffset = Vector2i{ 0 } );

#if 0
    // Data must be packed().
    // Only accepts RGB and BGR for now.
    bool set( Array2DView< const uint8x3 > data,
        GLImageFormat format = GLImageFormat::RGB,
        const Vector2i& dstOffset = Vector2i{ 0 } );

    // Data must be packed().
    // Only accepts RGBA and BGRA for now.
    bool set( Array2DView< const uint8x4 > data,
        GLImageFormat format = GLImageFormat::RGBA,
        const Vector2i& dstOffset = Vector2i{ 0 } );

    // Data must be packed().
    bool set( Array2DView< const float > data,
        GLImageFormat format = GLImageFormat::RED,
        const Vector2i& dstOffset = Vector2i{ 0 } );

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    // Also returns false if format isn't RGBA or BGRA.
    bool get( Array2DView< uint8x4 > output, GLImageFormat format = GLImageFormat::RGBA );

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    bool get( Array2DView< float > output );
    bool get( Array2DView< Vector2f > output );
    bool get( Array2DView< Vector4f > output );
#endif

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    // Also returns false if format isn't RGBA or BGRA.
    bool get( GLCubeMapFace face, Array2DWriteView< uint8x4 > output,
        GLImageFormat format = GLImageFormat::RGBA );

private:

    int m_sideLength;
};
