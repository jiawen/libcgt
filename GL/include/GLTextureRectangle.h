#pragma once

#include <GL/glew.h>

#include <common/Array2DView.h>
#include <common/BasicTypes.h>
#include <vecmath/Vector2i.h>

#include "GLTexture.h"

class Vector2f;
class Vector3f;
class Vector4f;

class GLTextureRectangle : public GLTexture
{
public:

	GLTextureRectangle( const Vector2i& size, GLImageInternalFormat internalFormat );

    // numElements = width * height
    int numElements() const;
    int width() const;
    int height() const;
    Vector2i size() const;

	// Uploads data to hardware.
    // Data must be linear.
    // TODO: do range checking, return a bool
    // TODO: the functions are almost all the same across formats,
    //   use a function that does the checking and finds the right formats.
    // TODO: make a version that converts?
    void set( Array2DView< const uint8_t > data );
    void set( Array2DView< const uint8x2 > data );
    void set( Array2DView< const uint8x3 > data );
    void set( Array2DView< const uint8x4 > data );
    void set( Array2DView< const float > data );
    void set( Array2DView< const Vector2f > data );
    void set( Array2DView< const Vector3f > data );
    void set( Array2DView< const Vector4f > data );

private:	

    Vector2i m_size;
};
