#include "Drawable.h"

GLDrawable::GLDrawable( GLPrimitiveType primitiveType,
    const PlanarVertexBufferCalculator& calculator ) :
    m_primitiveType( primitiveType ),
    m_calculator( calculator ),
    m_vbo( calculator.totalSizeBytes(),
        GLBufferObject::StorageFlags::MAP_WRITE_BIT )
{
    for( int i = 0; i < calculator.numAttributes(); ++i )
    {
        const PlanarVertexBufferCalculator::AttributeInfo& info =
            calculator.getAttributeInfo( i );

        m_vao.enableAttribute( i );
        if( info.type == GLVertexAttributeType::DOUBLE )
        {
            m_vao.setDoubleAttributeFormat
            (
                i,
                info.nComponents,
                0 /* relativeOffset. TODO: support this */
            );
        }
        else if( info.isInteger )
        {
            m_vao.setIntegerAttributeFormat
            (
                i,
                info.nComponents,
                info.type,
                0 /* relativeOffset. TODO: support this */
            );
        }
        else
        {
            m_vao.setAttributeFormat
            (
                i,
                info.nComponents,
                info.type,
                info.normalized,
                0 /* relativeOffset. TODO: support this */
            );
        }

        m_vao.attachBuffer( i, &m_vbo, info.offset,
            static_cast< GLsizei >( info.vertexStride ) );
    }
}

GLPrimitiveType GLDrawable::primitiveType() const
{
    return m_primitiveType;
}

int GLDrawable::numVertices() const
{
    return m_calculator.numVertices();
}

GLVertexArrayObject& GLDrawable::vao()
{
    return m_vao;
}


GLBufferObject& GLDrawable::vbo()
{
    return m_vbo;
}

void GLDrawable::draw()
{
    m_vao.bind();
    glDrawArrays( glPrimitiveType( m_primitiveType ), 0, numVertices() );
    GLVertexArrayObject::unbindAll();
}
