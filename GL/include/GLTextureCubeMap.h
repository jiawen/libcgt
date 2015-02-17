#pragma once

#include <GL/glew.h>

#include <common/Array2DView.h>
#include <common/BasicTypes.h>
#include <vecmath/Rect2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "GLTexture.h"

class GLSamplerObject;

class GLTextureCubeMap : public GLTexture
{
public:

    enum class Face
    {
        POSITIVE_X = GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        NEGATIVE_X = GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        POSITIVE_Y = GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        NEGATIVE_Y = GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        POSITIVE_Z = GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        NEGATIVE_Z = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    };
	
    GLTextureCubeMap( const Vector2i& size, GLImageInternalFormat internalFormat );
	
	int width() const;
	int height() const;
	Vector2i size() const;

    // TODO: clear with a rectangle
    void clear( const uint8x4& clearValue = uint8x4{ 0, 0, 0, 0 } );
    void clear( float clearValue = 0.f, GLImageFormat format = GLImageFormat::RED );
    void clear( const Vector4f& clearValue = Vector4f( 0, 0, 0, 0 ) );

    // Data must be packed().
    // Only accepts RGBA and BGRA for now.
    bool set( Face face, Array2DView< const uint8x4 > data,
		GLImageFormat format = GLImageFormat::RGBA,
		const Vector2i& dstOffset = Vector2i{ 0, 0 } );

#if 0
    // Data must be packed().
    // Only accepts RGB and BGR for now.
	bool set( Array2DView< const uint8x3 > data,
        GLImageFormat format = GLImageFormat::RGB,
		const Vector2i& dstOffset = Vector2i{ 0, 0 } );

    // Data must be packed().
    // Only accepts RGBA and BGRA for now.
    bool set( Array2DView< const uint8x4 > data,
		GLImageFormat format = GLImageFormat::RGBA,
		const Vector2i& dstOffset = Vector2i{ 0, 0 } );

    // Data must be packed().
    bool set( Array2DView< const float > data,
        GLImageFormat format = GLImageFormat::RED,
        const Vector2i& dstOffset = Vector2i{ 0, 0 } );

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
    bool get( Face face, Array2DView< uint8x4 > output,
        GLImageFormat format = GLImageFormat::RGBA );

private:

    Vector2i m_size;
};
