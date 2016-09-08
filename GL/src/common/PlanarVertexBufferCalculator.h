#pragma once

#include <vector>

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

    PlanarVertexBufferCalculator( int nVertices );

    // The immutable number of vertices for this calculator.
    int numVertices() const;

    // The number of attributes so far.
    int numAttributes() const;

    // The total size in bytes for the number of attributes so far.
    size_t totalSizeBytes() const;

    // TODO(jiawen): to support arbitrary formats, add a struct instead:
    // nComponents, GLVertexAttributeType type, bool normalized.
    // relativeOffset (bytes from beginning of a vertex where data starts) is 0
    // since this is planar.
    // TODO: can optionally add a string name and
    // a function to go between between index and name.

    // Add a generic vertex attribute, where for each vertex, it consists of
    // "nComponents" components of "componentSizeBytes" each.
    //
    // Returns the attribute index.
    int addAttribute( int nComponents, int componentSizeBytes );

    // The number of components for the i-th attribute.
    int numComponentsOf( int attributeIndex ) const;

    // The number of bytes for each component of the i-th attribute.
    int componentSizeOf( int attributeIndex ) const;

    // Number of bytes between vertices for the i-th attribute.
    // Equal to numComponentsOf( idx ) * componentSizeOf( idx ).
    int vertexSizeOf( int attributeIndex ) const;

    // The byte offset from the beginning of the vertex buffer where data for
    // the i-th attribute starts.
    int offsetOf( int attributeIndex ) const;

    // The number of bytes occupied by the data for the i-th attribute.
    int arraySizeOf( int attributeIndex ) const;

private:

    // TODO:
    // struct AttributeInfo
    //   string name
    //   int nComponents;
    //   int componentSize;
    //   int offset
    //   int arraySize;
    // next attribute index is just std::vector<AttributeInfo> size
    // total size is just calculated

    const int m_nVertices;

    // The number of components for the i-th attribute.
    // I.e., 1, 2, 3, or 4.
    std::vector< int > m_nComponents;

    // The size of each component for the i-th attribute
    // (e.g., sizeof( float ) ). If an attribute if "float3", this stores
    // sizeof( float ), not 3 * sizeof( float ).
    std::vector< int > m_componentSizes;

    // Since the data is planar, all elements of one attribute occupy one
    // contiguous array. Store its offsets from the beginning of the vertex
    // buffer and its size.
    // TODO(jiawen): allow an initial byte offset.

    // The byte offset from the beginning of the vertex buffer where data for
    // the i-th attribute starts.
    std::vector< int > m_offsets;

    // The number of bytes occupied by the data for the i-th attribute.
    std::vector< int > m_arraySizes;

    int m_nextAttributeIndex = 0;
    int m_totalSize = 0;
};
