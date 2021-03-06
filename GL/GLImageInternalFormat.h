#pragma once

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

enum class GLImageInternalFormat
{
    // Invalid, for default initialization.
    NONE = 0,

    // Color Formats.

    // 8-bit unsigned normalized integer
    R8 = GL_R8,
    RG8 = GL_RG8,
    RGB8 = GL_RGB8,
    RGBA8 = GL_RGBA8,

#ifdef GL_PLATFORM_ES_31
    BGRA8 = GL_BGRA8_EXT,
#endif

    // 8-bit signed integer
    R8I = GL_R8I,
    RG8I = GL_RG8I,
    RGB8I = GL_RGB8I,
    RGBA8I = GL_RGBA8I,

    // 8-bit unsigned integer
    R8UI = GL_R8UI,
    RG8UI = GL_RG8UI,
    RGB8UI = GL_RGB8UI,
    RGBA8UI = GL_RGBA8UI,

    // 16-bit signed integer
    R16I = GL_R16I,
    RG16I = GL_RG16I,
    RGB16I = GL_RGB16I,
    RGBA16I = GL_RGBA16I,

    // 16-bit unsigned integer
    R16UI = GL_R16UI,
    RG16UI = GL_RG16UI,
    RGB16UI = GL_RGB16UI,
    RGBA16UI = GL_RGBA16UI,

    // 32-bit signed integer
    R32I = GL_R32I,
    RG32I = GL_RG32I,
    RGB32I = GL_RGB32I,
    RGBA32I = GL_RGBA32I,

    // 32-bit unsigned integer
    R32UI = GL_R32UI,
    RG32UI = GL_RG32UI,
    RGB32UI = GL_RGB32UI,
    RGBA32UI = GL_RGBA32UI,

    // 16-bit float
    R16F = GL_R16F,
    RG16F = GL_RG16F,
    RGB16F = GL_RGB16F,
    RGBA16F = GL_RGBA16F,

    // 32-bit float
    R32F = GL_R32F,
    RG32F = GL_RG32F,
    RGB32F = GL_RGB32F,
    RGBA32F = GL_RGBA32F,

    // Depth formats.
    DEPTH_COMPONENT16 = GL_DEPTH_COMPONENT16,
    DEPTH_COMPONENT24 = GL_DEPTH_COMPONENT24,
#ifdef GL_PLATFORM_45
    DEPTH_COMPONENT32 = GL_DEPTH_COMPONENT32,
#endif
    DEPTH_COMPONENT32F = GL_DEPTH_COMPONENT32F,

    // depth + stencil
    DEPTH24_STENCIL8 = GL_DEPTH24_STENCIL8,
    DEPTH32F_STENCIL8 = GL_DEPTH32F_STENCIL8,

    // stencil only
    STENCIL_INDEX8 = GL_STENCIL_INDEX8,

#ifdef GL_PLATFORM_45
    // Compressed
    // unsigned normalized
    COMPRESSED_RED_RGTC1 = GL_COMPRESSED_RED_RGTC1,
    COMPRESSED_RG_RGTC2 = GL_COMPRESSED_RG_RGTC2,

    // signed normalized
    COMPRESSED_SIGNED_RED_RGTC1 = GL_COMPRESSED_SIGNED_RED_RGTC1,
    COMPRESSED_SIGNED_RG_RGTC2 = GL_COMPRESSED_SIGNED_RG_RGTC2,

    // TODO: HACK: getting compile errors on Windows with this for libcgt::cuda
    // only.
    // BPTC
    // unsigned normalized
    COMPRESSED_RGBA_BPTC_UNORM = GL_COMPRESSED_RGBA_BPTC_UNORM, // 4 component, RGBA
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM = GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM, // 4 component, sRGB, linear A

    // float, 3 components
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT = GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT,
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT = GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,

    // S3TC / DXT
    COMPRESSED_RGB_S3TC_DXT1_EXT = GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, // 1-bit alpha
    COMPRESSED_RGBA_S3TC_DXT3_EXT = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,
    COMPRESSED_RGBA_S3TC_DXT5_EXT = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,

    // S3TC in sRGB
    COMPRESSED_SRGB_S3TC_DXT1_EXT = GL_COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, // 1-bit alpha
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT = GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,
#endif

    // ----- Special formats -----

    // Normalized
#ifdef GL_PLATFORM_45
    R3_G3_B2 = GL_R3_G3_B2,
#endif
    RGB5_A1 = GL_RGB5_A1,
    RGB10_A2 = GL_RGB10_A2,

    // Integer
    RGB10_A2UI = GL_RGB10_A2UI,

    // Float
    R11F_G11F_B10F = GL_R11F_G11F_B10F,
    RGB9_E5 = GL_RGB9_E5,

    // sRGB
    SRGB8 = GL_SRGB8, // no alpha
    SRGB8_ALPHA8 = GL_SRGB8_ALPHA8, // linear alpha
};
