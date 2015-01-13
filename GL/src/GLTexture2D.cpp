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

void GLTexture2D::clear( const uint8x4& clearValue )
{
    glClearTexImage( id(), 0, GL_RGBA, GL_UNSIGNED_BYTE, &clearValue );
}

void GLTexture2D::clear( float clearValue, GLImageFormat format )
{
    glClearTexImage( id(), 0, static_cast< GLenum >( format ), GL_FLOAT,
        &clearValue );
}

void GLTexture2D::clear( const Vector4f& clearValue )
{
    glClearTexImage( id(), 0, GL_RGBA, GL_FLOAT, &clearValue );
}

bool GLTexture2D::set( Array2DView< const uint8x3 > data,
    GLImageFormat format,
    const Vector2i& dstOffset )
{
	if( dstOffset.x + data.width() > width() ||
		dstOffset.y + data.height() > height() )
	{
		return false;
	}
    if( format != GLImageFormat::RGB &&
		format != GLImageFormat::BGR )
	{
		return false;
	}

	glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
	// TODO: alignment, strides, ..., has to be packed: return false if not

	glTextureSubImage2DEXT( id(), GL_TEXTURE_2D, 0,
		dstOffset.x, dstOffset.y, data.width(), data.height(),
		static_cast< GLenum >( format ), GL_UNSIGNED_BYTE,
		data.pointer()
	);

	glPopClientAttrib();

	return true;
}

bool GLTexture2D::set( Array2DView< const uint8x4 > data,
	GLImageFormat format,
	const Vector2i& dstOffset )
{

	if( dstOffset.x + data.width() > width() ||
		dstOffset.y + data.height() > height() )
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

    glTextureSubImage2DEXT
    (
        id(), GL_TEXTURE_2D, 0,
        dstOffset.x, dstOffset.y, data.width(), data.height(),
        static_cast< GLenum >( format ), GL_UNSIGNED_BYTE,
        data.pointer()
    );

	glPopClientAttrib();

	return true;
}

bool GLTexture2D::set( Array2DView< const float > data,
	GLImageFormat format,
    const Vector2i& dstOffset )
{

	if( dstOffset.x + data.width() > width() ||
		dstOffset.y + data.height() > height() )
	{
		return false;
	}
	if( format != GLImageFormat::RED &&
		format != GLImageFormat::GREEN &&
        format != GLImageFormat::BLUE &&
        format != GLImageFormat::ALPHA )
	{
		return false;
	}

	glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
	// TODO: alignment, strides, ..., has to be packed

    glTextureSubImage2DEXT
    (
        id(), GL_TEXTURE_2D, 0,
        dstOffset.x, dstOffset.y, data.width(), data.height(),
        static_cast< GLenum >( format ), GL_FLOAT,
        data.pointer()
    );

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
	drawNV( Rect2f( 0, 0, static_cast< float >( width() ), static_cast< float >( height() ) ),
		pSampler, z, texCoords );
}

void GLTexture2D::drawNV( const Rect2f& windowCoords,
						 GLSamplerObject* pSampler,
						 float z,
						 const Rect2f& texCoords )
{
	GLuint samplerId = pSampler == nullptr ? 0 : pSampler->id();

    Vector2f p0 = windowCoords.origin();
    Vector2f p1 = windowCoords.limit();

    Vector2f t0 = texCoords.origin();
    Vector2f t1 = texCoords.limit();

    glDrawTextureNV( id(), samplerId,
        p0.x, p0.y,
        p1.x, p1.y,
		z,
        t0.x, t0.y,
        t1.x, t1.y
	);
}
