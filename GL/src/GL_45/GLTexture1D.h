#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/Array1DView.h>
#include <common/BasicTypes.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "GLTexture.h"
#include "GLPixelType.h"

class GLTexture1D : public GLTexture
{
public:

    // For a texture with dimensions baseSize, calculates the number of
    // mipmap levels needed, equal to 1 + floor(log(max(width))).
    static int calculateNumMipMapLevels( int baseSize );

    // For a texture with base dimensions baseSize, calculates the size
    // of a given level using the recursive formula:
    // nextLODdim = max(1, currentLODdim >> 1).
    // The base level is 0.
    static int calculateMipMapSizeForLevel( int baseSize, int level );

    // Allocate a 1D texture of the specified size, internalFormat, and number
    // of mipmap levels.
    //
    // If "nMipMapLevels" is set to a special value of 0, the number of levels
    // will be automatically calculated.
    GLTexture1D( int width, GLImageInternalFormat internalFormat,
                GLsizei nMipMapLevels = 1 );

    int numElements() const;
    int width() const;
    int size() const;

    // TODO: clear with a Range1i.
    // TODO: set: arbitrary format and void*?
    // TODO: relax the packing restriction.

    // srcFormat must be compatible with the texture's internal format.
    // srcData must be packed().
    bool set( Array1DView< const uint8_t > srcData,
             GLImageFormat srcFormat = GLImageFormat::RED,
             int dstOffset = 0 );
    bool set( Array1DView< const uint8x2 > srcData,
             GLImageFormat srcFormat = GLImageFormat::RG,
             int dstOffset = 0 );
    bool set( Array1DView< const uint8x3 > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGB,
             int dstOffset = 0 );
    bool set( Array1DView< const uint8x4 > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGBA,
             int dstOffset = 0 );
    bool set( Array1DView< const float > srcData,
             GLImageFormat srcFormat = GLImageFormat::RED,
             int dstOffset = 0 );
    bool set( Array1DView< const Vector2f > srcData,
             GLImageFormat srcFormat = GLImageFormat::RG,
             int dstOffset = 0 );
    bool set( Array1DView< const Vector3f > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGB,
             int dstOffset = 0 );
    bool set( Array1DView< const Vector4f > srcData,
             GLImageFormat srcFormat = GLImageFormat::RGBA,
             int dstOffset = 0 );

private:

    int m_width;

    bool checkSize( int srcSize, int dstOffset ) const;
    bool get1D( GLint srcLevel, GLImageFormat dstFormat, GLPixelType dstType,
               int dstSize, void* dstPtr );
    bool set1D( const void* srcPtr, int srcSize,
               GLImageFormat srcFormat, GLPixelType srcType, int dstOffset );
};
