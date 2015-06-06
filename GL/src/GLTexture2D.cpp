#include "GLTexture2D.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <GL/glew.h>

#include <common/Array2DView.h>
#include <math/Arithmetic.h>
#include <math/MathUtils.h>
#include <vecmath/Vector3f.h>

#include "GLSamplerObject.h"
#include "GLTexture2D.h"
#include "GLUtilities.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
int GLTexture2D::calculateNumMipMapLevels( const Vector2i& size )
{
    return 1 + Arithmetic::log2( MathUtils::maximum( size ) );
}

// static
Vector2i GLTexture2D::calculateMipMapSizeForLevel( const Vector2i& baseSize,
    int level )
{
    if( level <= 0 )
    {
        return baseSize;
    }

    Vector2i size = baseSize;
    while( level > 0 )
    {
        size = MathUtils::maximum( Vector2i{ 1, 1 }, size / 2 );
        --level;
    }
    return size;
}

GLTexture2D::GLTexture2D( const Vector2i& size, GLImageInternalFormat internalFormat,
    int nMipMapLevels ) :
    GLTexture( GL_TEXTURE_2D, internalFormat ),
    m_size( size )
{
    assert( size.x > 0 );
    assert( size.y > 0 );
    assert( size.x <= GLTexture::maxSize2D() );
    assert( size.y <= GLTexture::maxSize2D() );
    assert( nMipMapLevels >= 0 );

    // TODO: this can be put into GLTexture as well!
    if( nMipMapLevels == 0 )
    {
        m_nMipMapLevels = Arithmetic::log2( nMipMapLevels );
    }
    else
    {
        m_nMipMapLevels = nMipMapLevels;
    }

    glTextureStorage2D( id(), m_nMipMapLevels, static_cast< GLenum >( internalFormat ), size.x, size.y );
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

void GLTexture2D::clear( const uint8x4& clearValue, int level )
{
    glClearTexImage( id(), level, GL_RGBA, GL_UNSIGNED_BYTE, &clearValue );
}

void GLTexture2D::clear( float clearValue, GLImageFormat format, int level )
{
    glClearTexImage( id(), level, static_cast< GLenum >( format ), GL_FLOAT,
        &clearValue );
}

void GLTexture2D::clear( const Vector4f& clearValue, int level )
{
    glClearTexImage( id(), level, GL_RGBA, GL_FLOAT, &clearValue );
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

    // TODO(ARB_DSA): https://www.opengl.org/registry/specs/ARB/direct_state_access.txt
    // See 18): it was resolved to not deal with the pixel store bit.
	glPushClientAttribDefaultEXT( GL_CLIENT_PIXEL_STORE_BIT );
	// TODO: alignment, strides, ..., has to be packed: return false if not

	glTextureSubImage2D( id(), 0,
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

    glTextureSubImage2D
    (
        id(), 0,
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

    glTextureSubImage2D
    (
        id(), 0,
        dstOffset.x, dstOffset.y, data.width(), data.height(),
        static_cast< GLenum >( format ), GL_FLOAT,
        data.pointer()
    );

	glPopClientAttrib();

	return true;
}

bool GLTexture2D::get( Array2DView< uint8_t > output, GLImageFormat format )
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

    if( format != GLImageFormat::RED &&
        format != GLImageFormat::GREEN &&
        format != GLImageFormat::BLUE &&
        format != GLImageFormat::ALPHA )
    {
        return false;
    }
	
	// TODO: mipmap level
    // TODO: glGetTextureSubImage()
	glGetTextureImage( id(), 0,
		static_cast< GLenum >( format ), GL_UNSIGNED_BYTE, output.width() * output.height(), output );
	return true;
}

bool GLTexture2D::get( Array2DView< uint8x4 > output, GLImageFormat format )
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
	glGetTextureImage( id(), 0,
		static_cast< GLenum >( format ), GL_UNSIGNED_BYTE,
        output.width() * output.height() * output.elementStrideBytes(), output );
	return true;
}

bool GLTexture2D::get( Array2DView< float > output )					  
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
	glGetTextureImage( id(), 0,
		GL_RED, GL_FLOAT,
        output.width() * output.height() * output.elementStrideBytes(), output );
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
	glGetTextureImage( id(), 0,
		GL_RG, GL_FLOAT,
        output.width() * output.height() * output.elementStrideBytes(), output );
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
	glGetTextureImage( id(), 0,
		GL_RGBA, GL_FLOAT,
        output.width() * output.height() * output.elementStrideBytes(), output );
	return true;
}

void GLTexture2D::generateMipMaps()
{
    glGenerateTextureMipmap( id() );
}

void GLTexture2D::drawNV( GLSamplerObject* pSampler,
						 float z,
						 const Rect2f& texCoords )
{
	drawNV( Rect2f( size() ), pSampler, z, texCoords );
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
