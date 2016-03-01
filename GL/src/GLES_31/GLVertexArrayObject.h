#pragma once

#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

#include <common/Array1DView.h>
#include "GLVertexAttributeType.h"

class GLBufferObject;

/**
 * Be very aware of how index buffers work. Unlike the OpenGL 4.5 DSA version,
 * pre-DSA, it is illegal to bind an index buffer to the pipeline
 * (glBindBuffer(GL_ELEMENT_ARRAY_BUFFER)) without a VAO already bound. And this
 * binding lives *inside* the VAO itself. I.e., if you have two VAOs: this will
 * happen:
 *
 * GLVertexArrayObject vao0;
 * GLVertexArrayObject vao1;
 * GLBufferObject indexBuffer0;
 * GLBufferObject indexBuffer1;
 *
 * vao0.bind();
 * indexBuffer0.bind( GL_ELEMENT_ARRAY_BUFFER );
 *
 * vao1.bind();
 * indexBuffer1.bind( GL_ELEMENT_ARRAY_BUFFER );
 *
 * vao0.bind(); // indexBuffer0 is now the currently bound index buffer.
 *
 */
class GLVertexArrayObject
{
public:

    // Returns the id of the currently bound VAO.
    // Returns 0 if there is none.
    // (To get an actual GLVertexArrayObject object, you will have to make a
    // hash table).
    static GLuint boundId();

    // Unbinds all vertex array objects (only one can be bound at a time
    // anyway).
    static void unbindAll();

    // The maximum number of vertex attributes,
    // numbered 0 through maxNumVertexAttributes() - 1.
    // Corresponds to GL_MAX_VERTEX_ATTRIBS.
    static int maxNumVertexAttributes();

    // The maximum number of vertex buffer binding points,
    // numbered 0 through maxNumVertexAttribBindings() - 1.
    // Corresponds to GL_MAX_VERTEX_ATTRIB_BINDINGS.
    static int maxNumVertexAttributeBindings();

    // Maximum stride between elements in a buffer
    // when used as a vertex attribute.
    // Corresponds to GL_MAX_VERTEX_ATTRIB_STRIDE.
    static int maxVertexAttributeStride();

    GLVertexArrayObject();
    virtual ~GLVertexArrayObject();

    GLuint id() const;

    // Binds this vertex array object to the pipeline.
    void bind();

    // *If this VAO is bound to the pipeline*
    //
    // Enable the attribute at the given index. Corresponds to
    // glEnableVertexArrayAttrib().
    void enableAttribute( GLuint attributeIndex );

    // *If this VAO is bound to the pipeline*
    //
    // Enable the attribute at the given index. Corresponds to
    // glDisableVertexArrayAttrib().
    void disableAttribute( GLuint attributeIndex );

    // *If this VAO is bound to the pipeline*
    //
    // Associate attributeIndex with bindingIndex.
    // By default, the mapping is one-to-one:
    //   attribute 0 maps to binding 0.
    //   attribute 1 maps to binding 1.
    //   etc.
    //
    // It's useful to specify this mapping as many to one, especially for
    // interleaved formats. For example, if attribute 0 is POSITION, and
    // attribute 1 is NORMAL, they can both come from binding 0, which has an
    // interleaved buffer.
    void mapAttributeIndexToBindingIndex( GLuint attributeIndex,
        GLuint bindingIndex );

    // *If this VAO is bound to the pipeline*
    //
    // Set the format of an attribute.
    // nComponents: the number of components per elements (1, 2, 3, or 4).
    // type: the type of data: byte, int, float, etc.
    // normalized: whether or not to normalize fixed point data to float.
    //   If true and unsigned, 255 --> 1.0f, false: 255 --> 255.0f.
    //   Signed ranges map [-128, 127] --> [-1.0f, 1.0f].
    // relativeOffsetBytes: the number of bytes from the beginning of
    //   each vertex to look for this attribute. This is useful for interleaved
    //   formats.
    void setAttributeFormat( GLuint attributeIndex, GLint nComponents,
        GLVertexAttributeType type = GLVertexAttributeType::FLOAT,
        bool normalized = true, GLuint relativeOffsetBytes = 0 );

    // *If this VAO is bound to the pipeline*
    //
    // Set the format of an attribute.
    // nComponents: the number of components per elements (1, 2, 3, or 4).
    // type: the type of data: unsigned or singed byte, short, int, etc.
    // relativeOffsetBytes: the number of bytes from the beginning of
    //   each vertex to look for this attribute. This is useful for interleaved
    //   formats.
    void setAttributeIntegerFormat( GLuint attributeIndex, GLint nComponents,
        GLVertexAttributeType type, GLuint relativeOffsetBytes = 0 );

    // *If this VAO is bound to the pipeline*
    //
    // Attach a buffer to a binding index to be used as a vertex buffer.
    // offset is the number of bytes from the beginning of pBuffer's data
    //   where the vertex data starts.
    // stride is the number of bytes between the beginning of *entire vertices*
    //   and *cannot be 0*. It must also be <= maxVertexAttributeStride().
    //   As stated in
    //   https://www.opengl.org/wiki/Vertex_Specification: using the new API,
    //   the format is not known at attachment time, so OpenGL cannot compute
    //   it automatically.
    void attachBuffer( GLuint bindingIndex, GLBufferObject* pBuffer,
        GLintptr offset, GLsizei stride );

    // *If this VAO is bound to the pipeline*
    //
    // Detaches any buffer previously attached to the binding index.
    void detachBuffer( GLuint bindingIndex );

private:

    GLuint m_id;

};