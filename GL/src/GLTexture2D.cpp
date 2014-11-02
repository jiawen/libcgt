#include "GLTexture2D.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <GL/glew.h>

#include <common/Array2DView.h>
#include <vecmath/Vector3f.h>

#include "GLSamplerObject.h"
#include "GLTexture2D.h"
#include "GLUtilities.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

GLTexture2D::GLTexture2D( const Vector2i& size, GLImageInternalFormat internalFormat ) :
    GLTexture( GL_TEXTURE_2D, internalFormat ),
    m_size( size )
{
    assert( size.x > 0 );
    assert( size.y > 0 );
    assert( size.x <= GLTexture::maxSize1D2D() );
    assert( size.y <= GLTexture::maxSize1D2D() );

    glTextureStorage2DEXT( id(), GL_TEXTURE_2D, 1, static_cast< GLenum >( internalFormat ), size.x, size.y );
}

int GLTexture2D::width() const
{
    return m_size.x;
}

int GLTexture2D::height() const
{
    return m_size.y;
}

Vector2i GLTexture2D::size() const
{
    return m_size;
}

bool GLTexture2D::set( Array2DView< uint8x3 > data,
					  int xOffset, int yOffset )
{
	if( xOffset + data.width() > width() ||
		yOffset + data.height() > height() )
	{
		return false;
	}

	glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
	// TODO: alignment, strides, ..., has to be packed: return false if not

	glTextureSubImage2DEXT( id(), GL_TEXTURE_2D, 0,
		xOffset, yOffset, data.width(), data.height(),
		GL_RGB, GL_UNSIGNED_BYTE,
		data.pointer()
	);

	glPopClientAttrib();

	return true;
}

bool GLTexture2D::set( Array2DView< uint8x4 > data,
	GLImageFormat format,
	int xOffset, int yOffset )
{

	if( xOffset + data.width() > width() ||
		yOffset + data.height() > height() )
	{
		return false;
	}
	if( format != GLImageFormat::RGBA &&
		format != GLImageFormat::BGRA )
	{
		return false;
	}

	glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
	// TODO: alignment, strides, ..., has to be packed

    if( format == GLImageFormat::RGBA )
    {
        glTextureSubImage2DEXT
        (
            id(), GL_TEXTURE_2D, 0,
            xOffset, yOffset, data.width(), data.height(),
            GL_RGBA, GL_UNSIGNED_BYTE,
            data.pointer()
        );
    }
    else if( format == GLImageFormat::BGRA )
    {
        glTextureSubImage2DEXT
            (
            id(), GL_TEXTURE_2D, 0,
            xOffset, yOffset, data.width(), data.height(),
            GL_BGRA, GL_UNSIGNED_BYTE,
            data.pointer()
            );
    }

	glPopClientAttrib();

	return true;
}

bool GLTexture2D::get( Array2DView< uint8x4 > output )
{
	// TODO: glPixelStorei allows some packing?
    // GL_PACK_ALIGNMENT

	if( output.isNull() ||
		output.width() != width() ||
		output.height() != height() ||
		!( output.packed() ) )
	{
		return false;
	}
	
	// TODO: mipmap level
	glGetTextureImageEXT( id(), GL_TEXTURE_2D, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, output );
	return true;
}

bool GLTexture2D::get( Array2DView< Vector2f > output )					  
{
	// TODO: glPixelStorei allows some packing:
    // GL_PACK_ALIGNMENT

	if( output.isNull() ||
		output.width() != width() ||
		output.height() != height() ||
		!( output.packed() ) )
	{
		return false;
	}
	
	// TODO: GL_RG_INTEGER
	// output can be normalized or not
	// GL_RG_INTEGER for not normalized

	// TODO: level
	glGetTextureImageEXT( id(), GL_TEXTURE_2D, 0,
		GL_RG, GL_FLOAT, output );
	return true;
}

bool GLTexture2D::get( Array2DView< Vector4f > output )					  
{
	// TODO: glPixelStorei allows some packing?

	if( output.isNull() ||
		output.width() != width() ||
		output.height() != height() ||
		!( output.packed() ) )
	{
		return false;
	}
	
	// TODO: level
	glGetTextureImageEXT( id(), GL_TEXTURE_2D, 0,
		GL_RGBA, GL_FLOAT, output );
	return true;
}

void GLTexture2D::drawNV( GLSamplerObject* pSampler,
						 float z,
						 const Rect2f& texCoords )
{
	drawNV( Rect2f( 0, 0, width(), height() ),
		pSampler, z, texCoords );
}

void GLTexture2D::drawNV( const Rect2f& windowCoords,
						 GLSamplerObject* pSampler,
						 float z,
						 const Rect2f& texCoords )
{
	GLuint samplerId = pSampler == nullptr ? 0 : pSampler->id();
	glDrawTextureNV( id(), samplerId,
		windowCoords.left(), windowCoords.bottom(),
		windowCoords.right(), windowCoords.top(),
		z,
		texCoords.left(), texCoords.bottom(),
		texCoords.right(), texCoords.top()
	);
}
