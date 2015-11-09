#pragma once

#include <gl/glew.h>

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
    BGR = GL_BGR,
    BGRA = GL_BGRA,

    // Integer color formats.
    RED_INTEGER = GL_RED_INTEGER,
    GREEN_INTEGER = GL_GREEN_INTEGER,
    BLUE_INTEGER = GL_BLUE_INTEGER,
    ALPHA_INTEGER = GL_ALPHA_INTEGER,
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