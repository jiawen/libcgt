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

class GLTexture2D : public GLTexture
{
public:
	
    GLTexture2D( const Vector2i& size, GLImageInternalFormat internalFormat );
	
	int width() const;
	int height() const;
	Vector2i size() const;

    // TODO: clear with a rectangle
    void clear( const uint8x4& clearValue = uint8x4{ 0, 0, 0, 0 } );
    void clear( float clearValue = 0.f, GLImageFormat format = GLImageFormat::RED );
    void clear( const Vector4f& clearValue = Vector4f( 0, 0, 0, 0 ) );

	// TODO: set: arbitrary format and void*?

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

	bool get( Array2DView< Vector2f > output );
	bool get( Array2DView< Vector4f > output );	

	// Same as drawNV() below, but with
	// windowCoords = Rect2f( 0, 0, width(), height() )
	void drawNV( GLSamplerObject* pSampler = nullptr,
		float z = 0,
		const Rect2f& texCoords = Rect2f( 1.f, 1.f ) );

	// Using NV_draw_texture, draw this texture to the screen
	// to the rectangle windowCoords.
	//
	// The window will be mapped to have texCoords.
	// The default mapping draws right side up, OpenGL style.
	// To draw upside down, use Rect2f( x = 0, y = 1, width = 1, height = -1 ).
	// To draw a sub-rectangle, set texture coordinates between 0 and 1.
	//
	// Pass nullptr as the sampler object to use the default sampler
	// bound to the texture (NEAREST).
	//
	// The fragments will all have depth z.
	void drawNV( const Rect2f& windowCoords,
		GLSamplerObject* pSampler = nullptr,
		float z = 0,
		const Rect2f& texCoords = Rect2f( 1.f, 1.f ) );	

private:	

    Vector2i m_size;
};
