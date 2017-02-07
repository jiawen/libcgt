#pragma once

#include <GL/glew.h>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/GL/GLVertexAttributeType.h"

class GLBufferObject;

// http://us.download.nvidia.com/opengl/specs/GL_ARB_vertex_attrib_binding.txt
// - originally, there were (at least) 16 vertex attributes and 16 buffer bindings
//   with a fixed maping between them.
// - This extension changes the mapping.
// - glBindVertexBuffer() binds a vertex buffer to a binding index.
// - glVertexAttribBinding() / glVertexArrayAttribBinding() lets you bind
//   map attribute indices to binding indices.
// - glVertexAttribFormat() / glVertexArrayVertexAttribFormat().
//   - Describe the format of an attrib index.
//   - Replaces glVertexAttribPointer().
// glVertexArrayBindingDivisor(): only for instancing
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

    // TODO: can query for all the properties back:
    // glGetVertexArrayIndexediv() / glGetVertexArrayIndexed64iv()

    void bind();

    // Enable the attribute at the given index. Corresponds to
    // glEnableVertexArrayAttrib().
    void enableAttribute( GLuint attributeIndex );

    // Enable the attribute at the given index. Corresponds to
    // glDisableVertexArrayAttrib().
    void disableAttribute( GLuint attributeIndex );

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

    // Set the format of an attribute that will be interpreted as a float.
    // nComponents: the number of components per elements (1, 2, 3, or 4).
    // type: the type of data: byte, int, float, etc.
    // normalized: whether or not to normalize fixed point data to float.
    //   If true:
    //     Unsigned is mapped to [0, 1].
    //     Signed is mapped to [-1, 1].
    //   If false:
    //     value = static_cast< float >( src )
    // relativeOffsetBytes: the number of bytes from the beginning of
    //   each vertex to look for this attribute. This is useful for interleaved
    //   formats.
    void setAttributeFormat( GLuint attributeIndex, GLint nComponents,
        GLVertexAttributeType type = GLVertexAttributeType::FLOAT,
        bool normalized = true, GLuint relativeOffsetBytes = 0 );

    // Set the format of an attribute.
    // nComponents: the number of components per elements (1, 2, 3, or 4).
    // type: the type of data: unsigned or signed byte, short, int, etc.
    // relativeOffsetBytes: the number of bytes from the beginning of
    //   each vertex to look for this attribute. This is useful for interleaved
    //   formats.
    void setIntegerAttributeFormat( GLuint attributeIndex, GLint nComponents,
        GLVertexAttributeType type, GLuint relativeOffsetBytes = 0 );

    // Set the format of an attribute.
    // nComponents: the number of components per elements (1, 2, 3, or 4).
    // type is ignored: it's always double.
    // relativeOffsetBytes: the number of bytes from the beginning of
    //   each vertex to look for this attribute. This is useful for interleaved
    //   formats.
    void setDoubleAttributeFormat( GLuint attributeIndex, GLint nComponents,
        GLuint relativeOffsetBytes = 0 );

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

    // Attach count vertex buffers simultaneously. The three Array1DReadViews
    // must have the same size.
    void attachBuffers( GLuint firstBindingIndex,
        Array1DReadView< GLBufferObject* > buffers,
        Array1DReadView< GLintptr > offsets,
        Array1DReadView< GLsizei > strides );

    // Attach an index buffer (in OpenGL terminology, an element buffer).
    // There is only one slot.
    void attachIndexBuffer( GLBufferObject* pBuffer );

    // Gets the index of the attached index buffer.
    // Note: this VAO must have been bound to the pipeline at least once
    // or this will fail with GL_INVALID_OPERATION.
    // Corresponds to glGetVertexArrayiv();
    GLint getAttachedIndexBufferId();

    // Detach any buffer previously attached to the binding index.
    void detachBuffer( GLuint bindingIndex );

    // Detach any buffers attached to the range of binding indices.
    void detachBuffers( GLuint firstBindingIndex, int count );

private:

    GLuint m_id;

};


// TODO(ARB_DSA): query for current bindings
// // without DSA
// glGetVertexAttribiv
// GL_VERTEX_ARRAY_BINDING: which vertex array is bound (confirmed)
// GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: which buffer is bound to what slot (confirmed)
// GL_VERTEX_ATTRIB_BINDING: the mappping between  be the binding index
//    confirmed working with glGetVertexAttribiv(), but needs binding
//
// ARB_DSA:
// https://www.opengl.org/sdk/docs/man/html/glGetVertexArrayIndexed.xhtml
// glGetVertexArrayIndexediv() and glGetVertexArrayIndexed64iv()
