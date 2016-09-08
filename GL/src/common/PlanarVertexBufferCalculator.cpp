#include "PlanarVertexBufferCalculator.h"

PlanarVertexBufferCalculator::PlanarVertexBufferCalculator( int nVertices ) :
    m_nVertices( nVertices )
{

}

int PlanarVertexBufferCalculator::numVertices() const
{
    return m_nVertices;
}

int PlanarVertexBufferCalculator::numAttributes() const
{
    return static_cast< int >( m_offsets.size() );
}

size_t PlanarVertexBufferCalculator::totalSizeBytes() const
{
    return m_totalSize;
}

int PlanarVertexBufferCalculator::addAttribute( int nComponents,
    int componentSizeBytes )
{
    int attribIndex = m_nextAttributeIndex;
    int arraySize = m_nVertices * nComponents * componentSizeBytes;

    m_nComponents.push_back( nComponents );
    m_componentSizes.push_back( componentSizeBytes );
    m_offsets.push_back( m_totalSize );
    m_arraySizes.push_back( arraySize );

    m_totalSize += arraySize;
    ++m_nextAttributeIndex;

    return attribIndex;
}

int PlanarVertexBufferCalculator::numComponentsOf( int attributeIndex ) const
{
    return m_nComponents[ attributeIndex ];
}

int PlanarVertexBufferCalculator::componentSizeOf( int attributeIndex ) const
{
    return m_componentSizes[ attributeIndex ];
}

int PlanarVertexBufferCalculator::vertexSizeOf( int attributeIndex ) const
{
    return numComponentsOf( attributeIndex ) *
        componentSizeOf( attributeIndex );
}

int PlanarVertexBufferCalculator::offsetOf( int attributeIndex ) const
{
    return m_offsets[ attributeIndex ];
}

int PlanarVertexBufferCalculator::arraySizeOf( int attributeIndex ) const
{
    return m_arraySizes[ attributeIndex ];
}
