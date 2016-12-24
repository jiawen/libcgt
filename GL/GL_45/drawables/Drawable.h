#pragma once

#include "libcgt/core/common/ArrayView.h"

#include "libcgt/GL/GLPrimitiveType.h"
#include "libcgt/GL/PlanarVertexBufferCalculator.h"
#include "libcgt/GL/GL_45/GLBufferObject.h"
#include "libcgt/GL/GL_45/GLVertexArrayObject.h"

// TODO(jiawen): Make a GLDrawableSet class.
// Initialize it with a bunch of GLDrawable objects.
// It will pack everything into one large VBO.

// TODO(jiawen): Support glDrawElements()
class GLDrawable
{
public:

    // TODO(jiawen): Move this into GLBufferObject instead.
    // An RAII wrapper object around a mapped buffer.
    template< typename T >
    class MappedBuffer
    {
    public:

        Array1DWriteView< T > view()
        {
            return m_view;
        }

        ~MappedBuffer()
        {
            if( m_drawable != nullptr && m_view.notNull() )
            {
                m_drawable->m_vbo.unmap();
            }
        }

    private:
        friend class ::GLDrawable;
        GLDrawable* m_drawable;
        Array1DWriteView< T > m_view;

        MappedBuffer( GLDrawable* drawable, Array1DWriteView< T > view ) :
            m_drawable( drawable ),
            m_view( view )
        {

        }
    };

    GLDrawable( GLPrimitiveType primitiveType,
        const PlanarVertexBufferCalculator& calculator );

    GLPrimitiveType primitiveType() const;

    int numVertices() const;

    GLVertexArrayObject& vao();

    GLBufferObject& vbo();

    // TODO(jiawen): move this into GLVertexBufferObject itself.
    // Maps the range covered by attribute i for writing.
    // You may only map one attribute at a time.
    // TODO(jiawen): can you map more than one range at a time?
    template< typename T >
    MappedBuffer< T > mapAttribute( int i );

    // Binds the vertex array object, draws all vertices, and unbinds.
    void draw();

private:

    GLPrimitiveType m_primitiveType;
    PlanarVertexBufferCalculator m_calculator;
    GLVertexArrayObject m_vao;
    GLBufferObject m_vbo;
};

#include "Drawable.inl"
