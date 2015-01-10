#include "GLFramebufferObject.h"

#include "GLTexture.h"

#include <cassert>
#include <cstdio>

// ========================================
// Public
// ========================================

// static
GLuint GLFramebufferObject::boundId()
{
    int bid;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, &bid );
    return bid;
}

// static
void GLFramebufferObject::unbindAll()
{
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

// static
GLint GLFramebufferObject::getMaxColorAttachments()
{
	GLint maxColorAttachments;
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorAttachments );
	return maxColorAttachments;
}

GLFramebufferObject::GLFramebufferObject()
{
	glGenFramebuffers( 1, &m_id );
}

GLuint GLFramebufferObject::id() const
{
    return m_id;
}

// virtual
GLFramebufferObject::~GLFramebufferObject()
{
	glDeleteFramebuffers( 1, &m_id );
}

void GLFramebufferObject::bind()
{
	glBindFramebuffer( GL_FRAMEBUFFER, m_id );
}

void GLFramebufferObject::attachTexture( GLenum attachment, GLTexture* pTexture, GLint zSlice )
{
	// TODO: 1d texture?
	// rectangle can be target, not the same as type apparently
	// also cube maps

	GLuint textureId = pTexture->id();

	if( getAttachedId( attachment ) != textureId )
	{
		GLenum textureTarget = pTexture->target();
		switch( textureTarget )
		{
		// TODO:
		// http://www.opengl.org/wiki/Framebuffer_Object
		// to bind a mipmap level as a layered image, use glNamedFramebufferTextureEXT

		case GL_TEXTURE_2D:
			glNamedFramebufferTexture2DEXT( m_id, attachment,
				GL_TEXTURE_2D, textureId,
				0 ); // mipmap level
			break;

		case GL_TEXTURE_RECTANGLE:
			glNamedFramebufferTexture2DEXT( m_id, attachment,			
				GL_TEXTURE_RECTANGLE, textureId,
				0 ); // mipmap level
			break;

		// TODO: 3D and mipmap should just get their own functions
		case GL_TEXTURE_3D:
			glNamedFramebufferTexture3DEXT( m_id, attachment,
				GL_TEXTURE_3D, textureId,
				0, // mipmap level
				zSlice );			
			break;

		default:

			assert( false );
			break;
		}
	}	
}

void GLFramebufferObject::detachTexture( GLenum attachment )
{
	GLenum type = getAttachedType( attachment );
	switch( type )
	{
	case GL_NONE:
		break;
	case GL_RENDERBUFFER:
		glNamedFramebufferRenderbufferEXT( m_id, attachment, GL_RENDERBUFFER, 0 ); // 0 ==> detach		
		break;

	// TODO: 3D?
	case GL_TEXTURE:
		glNamedFramebufferTexture2DEXT( m_id, attachment,
			GL_TEXTURE_2D, // ignored, since detaching			
			0, // 0 ==> detach
			0 ); // mipmap level
		break;

	default:
		fprintf( stderr, "FramebufferObject::unbind_attachment ERROR: Unknown attached resource type\n" );
		assert( false );
		break;
	}
}

GLuint GLFramebufferObject::getAttachedId( GLenum attachment )
{
	GLint id;
	glGetNamedFramebufferAttachmentParameterivEXT( m_id, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &id );
	return id;
}

GLuint GLFramebufferObject::getAttachedType( GLenum attachment )
{
	GLint type = 0;
	glGetNamedFramebufferAttachmentParameterivEXT( m_id, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type );
	return type;
}

GLint GLFramebufferObject::getAttachedZSlice( GLenum attachment )
{
	GLint slice = -1;
	glGetNamedFramebufferAttachmentParameterivEXT( m_id, attachment,
		GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_EXT,
		&slice );
	return slice;
}

bool GLFramebufferObject::checkStatus( GLenum* pStatus )
{
	bool isComplete = false;	

	GLenum status = glCheckNamedFramebufferStatusEXT( m_id, GL_FRAMEBUFFER );
	switch( status )
	{
	case GL_FRAMEBUFFER_COMPLETE:
		// fprintf( stderr, "Framebuffer is complete.\n" );
		isComplete = true;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		fprintf( stderr, "Framebuffer incomplete: format unsupported.\n" );
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
		fprintf( stderr, "Framebuffer incomplete: incomplete attachment.\n" );
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
		fprintf( stderr, "Framebuffer incomplete: missing attachment.\n" );
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		fprintf( stderr, "Framebuffer incomplete: dimension mismatch.\n" );
		break;		
	case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		fprintf( stderr, "Framebuffer incomplete: incompatible formats.\n" );
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
		fprintf( stderr, "framebuffer INCOMPLETE_DRAW_BUFFER\n" );
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
		fprintf( stderr, "framebuffer INCOMPLETE_READ_BUFFER\n" );
		break;
	case GL_FRAMEBUFFER_BINDING:
		fprintf( stderr, "framebuffer BINDING\n" );
		break;
	default:
		fprintf( stderr, "Can't get here!\n" );
		assert( false );
	}

	if( pStatus != nullptr )
	{
		*pStatus = status;
	}
	return isComplete;
}
