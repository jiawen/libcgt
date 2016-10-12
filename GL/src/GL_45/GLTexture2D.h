#pragma once

#include <GL/glew.h>

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
    //
    // If "nMipMapLevels" is set to a special value of 0, the number of levels
    // will be automatically calculated.
    GLTexture2D();
    GLTexture2D( const Vector2i& size, GLImageInternalFormat internalFormat,
        int nMipMapLevels = 1 );
    GLTexture2D( const GLTexture2D& copy ) = delete;
    GLTexture2D( GLTexture2D&& move );
    GLTexture2D& operator = ( const GLTexture2D& copy ) = delete;
    GLTexture2D& operator = ( GLTexture2D&& move );
    virtual ~GLTexture2D();

    int numElements() const;
    int width() const;
    int height() const;
    Vector2i size() const;

    // TODO: clear with a rectangle.
    // TODO: get: integer formats.
    // TODO: document the conversions better.
    // TODO: set: arbitrary format and void*?
    // TODO: mipmap level

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    // dstFormat must be RED, GREEN, BLUE, or ALPHA.
    bool get( Array2DView< uint8_t > output,
             GLImageFormat dstFormat = GLImageFormat::RED );

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    bool get( Array2DView< uint8x2 > output );

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    // dstFormat must be RGB or BGR.
    bool get( Array2DView< uint8x3 > output,
             GLImageFormat dstFormat = GLImageFormat::RGB );

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    // dstFormat must be RGBA or BGRA.
    bool get( Array2DView< uint8x4 > output,
             GLImageFormat dstFormat = GLImageFormat::RGBA );

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    bool get( Array2DView< float > output,
             GLImageFormat dstFormat = GLImageFormat::RED );

    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    bool get( Array2DView< Vector2f > output );
    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    // dstFormat must be RGB or BGR.
    bool get( Array2DView< Vector3f > output,
             GLImageFormat dstFormat = GLImageFormat::RGB );
    // Retrieves the entire texture.
    // Returns false if output isNull(), is not packed, or has the wrong size.
    // dstFormat must be RGBA or BGRA.
    bool get( Array2DView< Vector4f > output,
             GLImageFormat dstFormat = GLImageFormat::RGBA );

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

    // Same as drawNV() below, but with
    // windowCoords = Rect2f( 0, 0, width(), height() )
    void drawNV( GLSamplerObject* pSampler = nullptr,
        float z = 0,
        const Rect2f& texCoords = Rect2f{ { 1, 1 } });

    // Using NV_draw_texture, draw this texture on screen scaled to the
    // rectangle windowCoords.
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
        const Rect2f& texCoords = Rect2f{ { 1, 1 } });

private:

    bool checkSize( const Vector2i& srcSize, const Vector2i& dstOffset ) const;
    bool get2D( GLint srcLevel,
               GLImageFormat dstFormat, GLPixelType dstType,
               const Vector2i& dstSize, void* dstPtr );
    bool set2D( const void* srcPtr, const Vector2i& srcSize,
               GLImageFormat srcFormat, GLPixelType srcType,
               const Vector2i& dstOffset );

    Vector2i m_size = { 0, 0 };
};
