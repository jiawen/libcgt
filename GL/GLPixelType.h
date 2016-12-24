#pragma once

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

enum class GLPixelType
{
    // Unsigned integers.
    UNSIGNED_BYTE = GL_UNSIGNED_BYTE,
    UNSIGNED_SHORT = GL_UNSIGNED_SHORT,
    UNSIGNED_INT = GL_UNSIGNED_INT,

    // Signed integers.
    BYTE = GL_BYTE,
    SHORT = GL_SHORT,
    INT = GL_INT,

    // TODO: Packed unsigned integers.
    // GL_UNSIGNED_BYTE_

    // Float.
    HALF_FLOAT = GL_HALF_FLOAT,
    FLOAT = GL_FLOAT,

    // Depth + Stencil.
    UNSIGNED_INT_24_8 = GL_UNSIGNED_INT_24_8,

    // 64-bit packed floating point depth + stencil.
    // [ 31 30 ...         0 | 31 30 ...         8 | 7 6 ... 0 ]
    // [ float part of depth | fixed part of depth | stencil   ]
    FLOAT_32_UNSIGNED_24_8_REV = GL_FLOAT_32_UNSIGNED_INT_24_8_REV
};

GLenum glPixelType( GLPixelType type );
