#include "libcgt/GL/PlanarVertexBufferCalculator.h"

#include <cassert>
#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

PlanarVertexBufferCalculator::PlanarVertexBufferCalculator( int nVertices ) :
    m_nVertices( nVertices )
{

}

int PlanarVertexBufferCalculator::numVertices() const
{
    return m_nVertices;
}

size_t PlanarVertexBufferCalculator::numAttributes() const
{
    return static_cast< size_t >( m_attributeInfo.size() );
}

size_t PlanarVertexBufferCalculator::totalSizeBytes() const
{
    size_t totalSize = 0;
    for( auto ai : m_attributeInfo )
    {
        totalSize += ai.arraySize;
    }
    return totalSize;
}

int PlanarVertexBufferCalculator::addAttribute( GLVertexAttributeType type,
    int nComponents, bool normalized, bool isInteger )
{
    // TODO: isValidCombination( vertexSizeBytes, nComponents ).
    if( nComponents < 0 || nComponents > 4 )
    {
        return -1;
    }
#ifdef GL_PLATFORM_45
    if( type == GLVertexAttributeType::DOUBLE )
    {
        return -1;
    }
#endif

    size_t vertexStride = vertexSizeBytes( type, nComponents );
    size_t arraySize = m_nVertices * vertexStride;

    m_attributeInfo.push_back
    (
        {
            isInteger,
            type,
            nComponents,
            normalized,
            totalSizeBytes(), // offset
            vertexStride,
            arraySize,
        }
    );

    return static_cast< int >( m_attributeInfo.size() );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< uint8_t >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_BYTE, 1, normalized );
}

// TODO: generic vector type
template<>
int PlanarVertexBufferCalculator::addAttribute< uint8x2 >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_BYTE, 2, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< uint8x3 >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_BYTE, 3, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< uint8x4 >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_BYTE, 4, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< uint16_t >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_SHORT, 1, normalized );
}

// TODO: generic vector type
template<>
int PlanarVertexBufferCalculator::addAttribute< uint16x2 >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_SHORT, 2, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< uint16x3 >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_SHORT, 3, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< uint16x4 >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::UNSIGNED_SHORT, 4, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< float >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::FLOAT, 1, normalized );
}

// TODO: generic vector type
template<>
int PlanarVertexBufferCalculator::addAttribute< Vector2f >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::FLOAT, 2, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< Vector3f >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::FLOAT, 3, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< Vector4f >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::FLOAT, 4, normalized );
}

template<>
int PlanarVertexBufferCalculator::addAttribute< double >( bool normalized )
{
    return addAttribute( GLVertexAttributeType::DOUBLE, 1, normalized );
}

const PlanarVertexBufferCalculator::AttributeInfo&
PlanarVertexBufferCalculator::getAttributeInfo( int index ) const
{
    assert( index < numAttributes() );
    return m_attributeInfo[ index ];
}
