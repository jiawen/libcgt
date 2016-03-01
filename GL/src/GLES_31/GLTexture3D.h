#pragma once

#include <cstdint>

#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

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

    static int calculateNumMipMapLevels( const Vector3i& size );
    static Vector3i calculateMipMapSizeForLevel( const Vector3i& baseSize,
            int level );

    GLTexture3D( const Vector3i& size, GLImageInternalFormat internalFormat,
                GLsizei nMipMapLevels = 1 );

    int numElements() const;
    int width() const;
    int height() const;
    int depth() const;
    Vector3i size() const;

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
    bool set3D( const void* srcPtr, const Vector3i& srcSize,
               GLImageFormat srcFormat, GLPixelType srcType,
               const Vector3i& dstOffset );

    Vector3i m_size;
};

