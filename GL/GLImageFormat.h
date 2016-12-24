#pragma once

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

enum class GLImageFormat
{
    // Floating point or impilicitly normalized color formats.
    RED = GL_RED,
    GREEN = GL_GREEN,
    BLUE = GL_BLUE,
    ALPHA = GL_ALPHA,
    RG = GL_RG,
    RGB = GL_RGB,
    RGBA = GL_RGBA,

    // Components reversed.
#ifdef GL_PLATFORM_45
    BGR = GL_BGR,
    BGRA = GL_BGRA,
#endif

#ifdef GL_PLATFORM_ES_31
    BGRA = GL_BGRA_EXT,
#endif

    // Integer color formats.
    RED_INTEGER = GL_RED_INTEGER,
#ifdef GL_PLATFORM_45
    GREEN_INTEGER = GL_GREEN_INTEGER,
    BLUE_INTEGER = GL_BLUE_INTEGER,
    ALPHA_INTEGER = GL_ALPHA_INTEGER,
#endif
    RG_INTEGER = GL_RG_INTEGER,
    RGB_INTEGER = GL_RGB_INTEGER,
    RGBA_INTEGER = GL_RGBA_INTEGER,

    // Depth only.
    DEPTH_COMPONENT = GL_DEPTH_COMPONENT,

    // Stencil only.
    STENCIL_INDEX = GL_STENCIL_INDEX,

    // Packed depth/stencil.
    DEPTH_STENCIL = GL_DEPTH_STENCIL
};

GLenum glImageFormat( GLImageFormat format );
