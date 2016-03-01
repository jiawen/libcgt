#pragma once

#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

#include <common/Array2DView.h>
#include <common/BasicTypes.h>
#include <vecmath/Rect2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "GLPixelType.h"
#include "GLTexture.h"

class GLSamplerObject;

class GLTexture2D : public GLTexture
{
public:

    // For a texture with dimensions baseSize, calculates the number of
    // mipmap levels needed, equal to 1 + floor(log(max(width, height))).
    static int calculateNumMipMapLevels( const Vector2i& baseSize );

    // For a texture with base dimensions baseSize, calculates the size
    // of a given level using the recursive formula:
    // nextLODdim = max(1, currentLODdim >> 1).
    // The base level is 0.
    static Vector2i calculateMipMapSizeForLevel( const Vector2i& baseSize,
        int level );

    // Allocate a 2D texture of the specified size, internalFormat, and number
    // of mipmap levels.
    // If "nMipMapLevels" is set to a special value of 0, the number of levels will
    // be automatically calculated.
    GLTexture2D( const Vector2i& size, GLImageInternalFormat internalFormat,
        GLsizei nMipMapLevels = 1 );

    int width() const;
    int height() const;
    Vector2i size() const;

    // TODO: set: arbitrary format and void*?
    // TODO: relax the packing restriction.

    // dstOffset + dstData.size must be < size().
    // srcFormat must be compatible with the texture's internal format.
    // srcData must be packed().
    bool set( Array2DView< const uint8_t > srcData,
             GLImageFormat srcFormat = GLImageFormat::RED,
             const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DView< const uint8x2 > srcData,
             GLImageFormat srcFormat = GLImageFormat::RG,
             const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DView< const uint8x3 > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGB,
             const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DView< const uint8x4 > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGBA,
             const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DView< const float > srcData,
             GLImageFormat srcFormat = GLImageFormat::RED,
             const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DView< const Vector2f > srcData,
             GLImageFormat srcFormat = GLImageFormat::RG,
             const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DView< const Vector3f > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGB,
             const Vector2i& dstOffset = { 0, 0 } );
    bool set( Array2DView< const Vector4f > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGBA,
             const Vector2i& dstOffset = { 0, 0 } );


private:

    bool checkSize( const Vector2i& srcSize, const Vector2i& dstOffset ) const;
    bool set2D( const void* srcPtr, const Vector2i& srcSize,
               GLImageFormat srcFormat, GLPixelType srcType,
               const Vector2i& dstOffset );

    Vector2i m_size;
};
