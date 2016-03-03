#pragma once

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

// Types that can be used as a vertex attribute, for:
// glVertexAttribFormat / glVertexAttribIFormat / glVertexAttribLFormat
// and their vertex array equivalents.
// Mostly used when calling GLVertexArrayObject::setAttributeFormat().
enum class GLVertexAttributeType
{
    // Available to both glVertexAttribFormat() and glVertexAttribIFormat()
    UNSIGNED_BYTE = GL_UNSIGNED_BYTE,
    UNSIGNED_SHORT = GL_UNSIGNED_SHORT,
    UNSIGNED_INT = GL_UNSIGNED_INT,

    BYTE = GL_BYTE,
    SHORT = GL_SHORT,
    INT = GL_INT,

    // Also available to glVertexAttribFormat(), but not glVertexAttribIFormat()
    FIXED = GL_FIXED,

    HALF_FLOAT = GL_HALF_FLOAT,
    FLOAT = GL_FLOAT,

    // 4 signed integers packed into a single 32-bit quantity.
    INT_2_10_10_10_REV = GL_INT_2_10_10_10_REV,

    // 4 unsigned integers packed intoa single 32-bit quantity.
    UNSIGNED_INT_2_10_10_10_REV = GL_INT_2_10_10_10_REV,

#ifdef GL_PLATFORM_45
    // For glVertexAttribLFormat() exclusively.
    // glVertexAttribLFormat() can't use a different format.
    DOUBLE = GL_DOUBLE
#endif
};

GLenum glVertexAttributeType( GLVertexAttributeType type );