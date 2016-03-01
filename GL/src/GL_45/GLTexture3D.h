#pragma once

#include <cstdint>

#include <GL/glew.h>

#include <common/Array3DView.h>
#include <common/BasicTypes.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

#include "GLPixelType.h"
#include "GLTexture.h"

class GLTexture3D : public GLTexture
{
public:

    // For a texture with dimensions baseSize, calculates the number of
    // mipmap levels needed, equal to 1 + floor(log(max(width, height, depth))).
    static int calculateNumMipMapLevels( const Vector3i& size );

    // For a texture with base dimensions baseSize, calculates the size
    // of a given level using the recursive formula:
    // nextLODdim = max(1, currentLODdim >> 1).
    // The base level is 0.
    static Vector3i calculateMipMapSizeForLevel( const Vector3i& baseSize,
                                                int level );

    // Allocate a 3D texture of the specified size, internalFormat, and number
    // of mipmap levels.
    //
    // If "nMipMapLevels" is set to a special value of 0, the number of levels
    // will be automatically calculated.
    GLTexture3D( const Vector3i& size, GLImageInternalFormat internalFormat,
                GLsizei nMipMapLevels = 1 );

    int numElements() const;
    int width() const;
    int height() const;
    int depth() const;
    Vector3i size() const;

    // TODO: clear with a Box3i.

    // TODO: set: arbitrary format and void*?
    // TODO: relax the packing restriction.

    // Data must be packed().
    bool set( Array3DView< const uint8_t > data,
             GLImageFormat format = GLImageFormat::RED,
             const Vector3i& dstOffset = { 0, 0, 0 } );
    bool set( Array3DView< const uint8x2 > data,
             GLImageFormat format = GLImageFormat::RG,
             const Vector3i& dstOffset = { 0, 0, 0 } );
    bool set( Array3DView< const uint8x3 > data,
             GLImageFormat format = GLImageFormat::RGB,
             const Vector3i& dstOffset = { 0, 0, 0 } );
    bool set( Array3DView< const uint8x4 > data,
             GLImageFormat format = GLImageFormat::RGBA,
             const Vector3i& dstOffset = { 0, 0, 0 } );
    bool set( Array3DView< const float > data,
             GLImageFormat format = GLImageFormat::RED,
             const Vector3i& dstOffset = { 0, 0, 0 } );
    bool set( Array3DView< const Vector2f > data,
             GLImageFormat format = GLImageFormat::RG,
             const Vector3i& dstOffset = { 0, 0, 0 } );
    bool set( Array3DView< const Vector3f > data,
             GLImageFormat format = GLImageFormat::RGB,
             const Vector3i& dstOffset = { 0, 0, 0 } );
    bool set( Array3DView< const Vector4f > data,
             GLImageFormat format = GLImageFormat::RGBA,
             const Vector3i& dstOffset = { 0, 0, 0 } );

private:

    bool checkSize( const Vector3i& srcSize, const Vector3i& dstOffset ) const;
    bool get3D( GLint srcLevel,
               GLImageFormat dstFormat, GLPixelType dstType,
               const Vector3i& dstSize, void* dstPtr );
    bool set3D( const void* srcPtr, const Vector3i& srcSize,
               GLImageFormat srcFormat, GLPixelType srcType,
               const Vector3i& dstOffset );

    Vector3i m_size;
};
