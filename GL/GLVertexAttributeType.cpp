#include "GLVertexAttributeType.h"

#include <cassert>

// TODO: consider inlining these.
// TODO: C++14 may allow these to be converted to constexpr.

GLenum glVertexAttributeType( GLVertexAttributeType type )
{
    return static_cast< GLenum >( type );
}

size_t vertexSizeBytes( GLVertexAttributeType type, int nComponents )
{
    switch( type )
    {
    case GLVertexAttributeType::UNSIGNED_BYTE:
        return nComponents * sizeof( GLubyte );
    case GLVertexAttributeType::UNSIGNED_SHORT:
        return nComponents * sizeof( GLushort );
    case GLVertexAttributeType::UNSIGNED_INT:
        return nComponents * sizeof( GLuint );

    case GLVertexAttributeType::BYTE:
        return nComponents * sizeof( GLbyte );
    case GLVertexAttributeType::SHORT:
        return nComponents * sizeof( GLshort );
    case GLVertexAttributeType::INT:
        return nComponents * sizeof( GLint );

    case GLVertexAttributeType::FIXED:
        return nComponents * sizeof( GLfixed );

    case GLVertexAttributeType::HALF_FLOAT:
        return nComponents * sizeof( GLhalf );
    case GLVertexAttributeType::FLOAT:
        return nComponents * sizeof( GLfloat );

    case GLVertexAttributeType::INT_2_10_10_10_REV:
        assert( nComponents == 4 );
        return sizeof( GLint );

    case GLVertexAttributeType::UNSIGNED_INT_2_10_10_10_REV:
        assert( nComponents == 4 );
        return sizeof( GLuint );

#ifdef GL_PLATFORM_45
    case GLVertexAttributeType::UNSIGNED_INT_10F_11F_11F_REV:
        assert( nComponents == 3 );
        return sizeof( GLuint );

    case GLVertexAttributeType::DOUBLE:
        return nComponents * sizeof( GLdouble );
#endif

    default:
        return 0;
    }
}

bool isValidIntegerType( GLVertexAttributeType type )
{
    return type == GLVertexAttributeType::UNSIGNED_BYTE ||
        type == GLVertexAttributeType::UNSIGNED_SHORT ||
        type == GLVertexAttributeType::UNSIGNED_INT ||
        type == GLVertexAttributeType::BYTE ||
        type == GLVertexAttributeType::SHORT ||
        type == GLVertexAttributeType::INT;
}
