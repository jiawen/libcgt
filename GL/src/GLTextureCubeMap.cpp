#include "GLTextureCubeMap.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <GL/glew.h>

#include <common/Array2DView.h>
#include <vecmath/Vector3f.h>

#include "GLSamplerObject.h"
#include "GLTextureCubeMap.h"
#include "GLUtilities.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

GLTextureCubeMap::GLTextureCubeMap( const Vector2i& size, GLImageInternalFormat internalFormat ) :
    GLTexture( GL_TEXTURE_CUBE_MAP, internalFormat ),
    m_size( size )
{
    assert( size.x > 0 );
    assert( size.y > 0 );
    assert( size.x <= GLTexture::maxSizeCubeMap() );
    assert( size.y <= GLTexture::maxSizeCubeMap() );

    glTextureStorage2DEXT( id(), target(), 1, static_cast< GLenum >( internalFormat ), size.x, size.y );
}

int GLTextureCubeMap::width() const
{
    return m_size.x;
}

int GLTextureCubeMap::height() const
{
    return m_size.y;
}

Vector2i GLTextureCubeMap::size() const
{
    return m_size;
}

void GLTextureCubeMap::clear( const uint8x4& clearValue )
{
    glClearTexImage( id(), 0, GL_RGBA, GL_UNSIGNED_BYTE, &clearValue );
}

void GLTextureCubeMap::clear( float clearValue, GLImageFormat format )
{
    glClearTexImage( id(), 0, static_cast< GLenum >( format ), GL_FLOAT,
        &clearValue );
}

void GLTextureCubeMap::clear( const Vector4f& clearValue )
{
    glClearTexImage( id(), 0, GL_RGBA, GL_FLOAT, &clearValue );
}

bool GLTextureCubeMap::set( GLTextureCubeMap::Face face,
    Array2DView< const uint8x4 > data,
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
        id(), static_cast< GLenum >( face ), 0,
        dstOffset.x, dstOffset.y, data.width(), data.height(),
        static_cast< GLenum >( format ), GL_UNSIGNED_BYTE,
        data.pointer()
    );

	glPopClientAttrib();

	return true;
}

#if 0
bool GLTextureCubeMap::set( Array2DView< const uint8x3 > data,
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

bool GLTextureCubeMap::set( Array2DView< const uint8x4 > data,
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

bool GLTextureCubeMap::set( Array2DView< const float > data,
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

bool GLTextureCubeMap::get( Array2DView< uint8x4 > output, GLImageFormat format )
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

    if( format != GLImageFormat::RGBA &&
        format != GLImageFormat::BGRA )
    {
        return false;
    }
	
	// TODO: mipmap level
	glGetTextureImageEXT( id(), GL_TEXTURE_2D, 0,
		static_cast< GLenum >( format ), GL_UNSIGNED_BYTE, output );
	return true;
}


bool GLTextureCubeMap::get( Array2DView< float > output )					  
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
		GL_RED, GL_FLOAT, output );
	return true;
}

bool GLTextureCubeMap::get( Array2DView< Vector2f > output )					  
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

bool GLTextureCubeMap::get( Array2DView< Vector4f > output )					  
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
#endif


bool GLTextureCubeMap::get( GLTextureCubeMap::Face face, Array2DView< uint8x4 > output, GLImageFormat format )
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

    if( format != GLImageFormat::RGBA &&
        format != GLImageFormat::BGRA )
    {
        return false;
    }
	
	// TODO: mipmap level
	glGetTextureImageEXT( id(), static_cast< GLenum >( face ), 0,
		static_cast< GLenum >( format ), GL_UNSIGNED_BYTE, output );
	return true;
}
