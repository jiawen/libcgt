#pragma once

#include <GL/glew.h>

class GLVertexArrayObject
{
public:

	static void unbind();

	GLVertexArrayObject();
	virtual ~GLVertexArrayObject();

	GLuint id();

	// TODO:
	// once GLEW is more mature, wrt EXT_direct_state_access:
	// (GLEW 1.10.2)
	// http://sourceforge.net/p/glew/bugs/195/
	// glVertexArrayBindVertexBufferEXT()
	// glVertexArrayVertexAttribBindingEXT()
	// glVertexArrayVertexAttribFormatEXT()
	// glEnableVertexArrayAttribEXT()
	
	// glVertexAttribPointer: once in shader-land,
	//   the input is interpreted as floats no matter what.
	//   (If normalization is on, gets turned into [0,1] or [-1,1]).
	// glVertexAttribIPointer: input is always left as an integer.
	void bind();
	
	void enableAttribute( GLuint index );
	void disableAttribute( GLuint index );

private:

	GLuint m_id;

};
