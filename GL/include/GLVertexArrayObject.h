#pragma once

#include <GL/glew.h>

#include "GLVertexAttributeType.h"

class GLBufferObject;

// http://us.download.nvidia.com/opengl/specs/GL_ARB_vertex_attrib_binding.txt
    // - originally, there were (at least) 16 vertex attributes and 16 buffer bindings
    //   with a fixed maping between them.
    // - This extension changes the mapping.
	// - glVertexAttribFormat() glVertexArrayVertexAttribFormatEXT() / glVertexArrayVertexAttribFormat()
    //   - Describe the format of an attrib index.
    //   - Replaces glVertexAttribPointer
    // glVertexArrayBindingDivisor(): only for instancing
class GLVertexArrayObject
{
public:

    // Returns the id of the currently bound VAO.
    // Returns 0 if there is none.
    // (to get an actual GLVertexArrayObject object, you will have to main
    // a hash table).
    static GLuint boundId();

    // Unbinds all vertex array objects
    // (only one can be bound at a time anyway).
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

	// TODO:
	// once GLEW is more mature, wrt EXT_direct_state_access:
	// (still broken as of GLEW 1.11.1).
	// http://sourceforge.net/p/glew/bugs/195/
	
    // TODO: can query for all the properties back:
    // glGetVertexArrayIndexediv() / glGetVertexArrayIndexed64iv()

	void bind();

    // Corresponds to glEnableVertexArrayAttribEXT() and glDisableVertexArrayAttribEXT().
	void enableAttribute( GLuint attributeIndex );
	void disableAttribute( GLuint attributeIndex );

    // Associate attributeIndex with bindingIndex.
    // By default, the mapping is one-to-one:
    //   attribute 0 maps to binding 0.
    //   attribute 1 maps to binding 1.
    //   etc.
    //
    // It's useful to specify this mapping as many to one, especially for interleaved formats:
    // For example, if attribute 0 is POSITION, and attribute 1 is NORMAL,
    // they can both come from binding 0.
    // TODO: default? call get() to find out.
    void mapAttributeIndexToBindingIndex( GLuint attributeIndex,
        GLuint bindingIndex );

    // GLVertexAttributeType,
    // Set the format of an attribute.
    // nComponents: the number of components per elements (1, 2, 3, or 4).
    // type: the type of data.
    // normalized: whether or not to normalize fixed point data to float.
    //   If true and unsigned, 255 --> 1.0f, false: 255 --> 255.0f.
    //   Signed ranges map [-128, 127] --> [-1.0f, 1.0f].
    // relativeOffsetBytes: for vertex 0, the number of bytes from the
    //   beginning of the buffer to look for this attribute.
    //   This is useful for interleaved formats.
    void setAttributeFormat( GLuint attributeIndex, GLint nComponents,
        GLVertexAttributeType type = GLVertexAttributeType::FLOAT,
        bool normalized = true, GLuint relativeOffsetBytes = 0 );
    void setAttributeIntegerFormat( GLuint attributeIndex, GLint nComponents,
        GLVertexAttributeType type, GLuint relativeOffsetBytes = 0 );
    void setAttributeDoubleFormat( GLuint attributeIndex, GLint nComponents,
        GLuint relativeOffsetBytes = 0 );
    
	// glVertexArrayBindVertexBufferEXT() / glVertexArrayVertexBuffer()
    //   - with direct_state_access, directly bind a buffer to
    //     this VAO at index bindingIndex (with a buffer offset and stride)
    // stride is the byte offset between the beginning of *entire vertices* and
    //   *cannot be 0*, unlike some parts of the documentation.
    void attachBuffer( GLuint bindingIndex, GLBufferObject* pBuffer,
        GLintptr offset, GLsizei stride );
    void detachBuffer( GLuint bindingIndex );

    // TODO(ARB_DSA): element array buffers (aka index buffer)
    // are not supported in EXT_direct_state_access. Need ARB_DSA.
    // The call would be: glVertexArrayElementBuffer()
    // http://stackoverflow.com/questions/3776726/how-to-bind-a-element-buffer-array-to-vertex-array-object-using-direct-state-a

private:

	GLuint m_id;

};


// TODO: query for current bindings
// // without DSA
// glGetVertexAttribiv

// GL_VERTEX_ARRAY_BINDING: which vertex array is bound (confirmed)
// GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: which buffer is bound to what slot (confirmed)
// GL_VERTEX_ATTRIB_BINDING: the mappping between  be the binding index
//    confirmed working with glGetVertexAttribiv(), but needs binding
//
// ARB_DSA:
// glGetVertexArrayIndexediv() and glGetVertexArrayIndexed64iv()
//
// Should work but doesn't:
// glGetVertexArrayIntegeri_vEXT( m_pVAO->id(), 0, GL_VERTEX_ATTRIB_BINDING, &binding0 );
// glGetVertexArrayIntegeri_vEXT( m_pVAO->id(), 1, GL_VERTEX_ATTRIB_BINDING, &binding1 );