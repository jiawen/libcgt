#pragma once

#include <GL/glew.h>

#include <common/ArrayView.h>
#include <common/BasicTypes.h>
#include <vecmath/Vector2i.h>

#include "GLPixelType.h"
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

    // srcFormat must be compatible with the texture's internal format.
    // srcData must be packed().
    bool set( Array2DReadView< uint8_t > srcData,
        GLImageFormat srcFormat = GLImageFormat::RED,
        const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DReadView< uint8x2 > srcData,
        GLImageFormat srcFormat = GLImageFormat::RG,
        const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DReadView< uint8x3 > srcData,
        GLImageFormat srcFormat = GLImageFormat::RGB,
        const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DReadView< uint8x4 > srcData,
        GLImageFormat srcFormat = GLImageFormat::RGBA,
        const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DReadView< float > srcData,
        GLImageFormat srcFormat = GLImageFormat::RED,
        const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DReadView< Vector2f > srcData,
        GLImageFormat srcFormat = GLImageFormat::RG,
        const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DReadView< Vector3f > srcData,
        GLImageFormat srcFormat = GLImageFormat::RGB,
        const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DReadView< Vector4f > srcData,
        GLImageFormat srcFormat = GLImageFormat::RGBA,
        const Vector2i& dstOffset = { 0, 0 } );

private:

    Vector2i m_size;

    // TODO: make a GLenum getGLType( Array2DReadView< T > );
    bool checkSize( const Vector2i& srcSize, const Vector2i& dstOffset ) const;
    bool set2D( const void* srcPtr, const Vector2i& srcSize,
        GLImageFormat srcFormat, GLPixelType srcType,
        const Vector2i& dstOffset );
};
