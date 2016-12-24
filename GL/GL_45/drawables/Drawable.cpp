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
        m_vao.enableAttribute( i );
        // HACK: type should be stored
        auto type = GLVertexAttributeType::FLOAT;
        m_vao.setAttributeFormat
            (
            i,
            calculator.numComponentsOf( i ),
            type,
            false, /* normalized */
            0 /* relativeOffset */
            );

        m_vao.attachBuffer( i, &m_vbo, calculator.offsetOf( i ),
            calculator.vertexSizeOf( i ) );
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
