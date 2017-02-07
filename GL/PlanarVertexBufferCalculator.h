#pragma once

#include <vector>

#include "libcgt/GL/GLVertexAttributeType.h"

// TODO(jiawen):
// Planar vertex buffers are simple:
// - There are always exactly n vertices.
// - Since it's planar, every time you add an attribute, you append a
//   fixed-size segment to the end of the buffer.
//
// Interleaved is more annoying - you can have different binding indices, etc.
//
// Index buffers:
// - You have a fixed number of indices, which is the number of vertices to
//   draw.
// - You may likely have a different number of data items, but is the same
//   for each attribute.

// The data must be planar and will be packed into a single buffer object.
class PlanarVertexBufferCalculator
{
public:

    struct AttributeInfo
    {
        // If true, fixed-point values passed to the shader directly as
        // integers without conversion to float. Otherwise, they are converted
        // to float using either normalization or cast (determined by the
        // "normalized" field).
        //
        // isValidIntegerType( type ) must be true.
        bool isInteger;

        // Data type.
        GLVertexAttributeType type;

        // The number of components (1, 2, 3, or 4).
        int nComponents;

        // If type is not HALF_FLOAT, FLOAT, or DOUBLE, and isInteger is false,
        // determines whether fixed-point data is normalized
        // (unsigned to [0, 1], signed to [-1, 1]).
        bool normalized;

        // The offset from the beginning of the vertex buffer where data
        // for this attribute starts.
        size_t offset;

        // The number of bytes between vertices for this attribute.
        // Equal to nComponents * componentSize.
        // TODO: support relativeOffset, which is the number of bytes from the
        // beginning of *each vertex* where the data is sourced.
        // I.e., the data is stored WXYZ and you only want XYZ. You can set
        // relativeOffset to 4 bytes (1 float) and vertex stride to 16 bytes.
        size_t vertexStride;

        // The total number of bytes occupied by all the data for this
        // attribute.
        size_t arraySize;
    };

    PlanarVertexBufferCalculator( int nVertices );

    // The immutable number of vertices for this calculator.
    int numVertices() const;

    // The number of attributes added so far.
    size_t numAttributes() const;

    // The total size in bytes for the number of attributes so far.
    size_t totalSizeBytes() const;

    // Add a generic vertex attribute.
    // Each vertex consists of a specific type (an enum), the number of
    // components (1, 2, 3 or 4), and whether it should be normalized.
    //
    // normalized:
    // - Ignored if isInteger is true or isValidIntegerType( type ) is false.
    // - If true, then unsigned fixed point types are normalized to [0, 1], and
    //   signed types are normalized to [-1, 1].
    // - If false, then fixed point values are directly cast to float.
    //
    // isInteger:
    // - Invalid for any
    //
    // If the parameter combination is valid, returns the attribute index.
    // Otherwise, returns -1.
    //
    // TODO: add vertexStride
    // TODO: implement isValidIntegerComponent
    int addAttribute( GLVertexAttributeType type, int nComponents,
        bool normalized = true, bool isInteger = false );

    // Add a generic vertex attribute for common types.
    //
    // Supported types:
    // uint8_t, uint8x2, uint8x3, uint8x4,
    // uint16_t, uint16x2, uint16x3, uint16x4,
    // float, Vector2f, Vector3f, Vector4f
    // double
    //
    // TODO: Vector< d, type >
    // TODO: signed types
    // TODO: 32-bit types
    // TODO: VertexAttributeType< T > and numComponents< T >
    template< typename T >
    int addAttribute( bool normalized = true );

    // TODO:
    // template< typename T >
    // int addIntegerAttribute();

    // Get the info for the index-th attribute added.
    const AttributeInfo& getAttributeInfo( int index ) const;

private:

    const int m_nVertices;

    std::vector< AttributeInfo > m_attributeInfo;
};
