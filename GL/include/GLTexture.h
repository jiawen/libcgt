#ifndef GL_TEXTURE_H
#define GL_TEXTURE_H

#include <cstdint>

#include <GL/glew.h>
#include <QString>

#include <common/BasicTypes.h>

class GLTexture
{
public:

	// TODO: make getNumComponents() work always (depending on setData())
	// getData() and dump() should automatically know	

	// TODO: mipmaps, anisotropic
	enum GLTextureFilterMode
	{
		GLTextureFilterMode_NEAREST = GL_NEAREST,
		GLTextureFilterMode_LINEAR = GL_LINEAR
	};

	enum GLTextureInternalFormat
	{
        // 8-bit unsigned normalized integer
        R_8_BYTE_UNORM = GL_R8,
        RG_8_BYTE_UNORM = GL_RG8,
        RGB_8_BYTE_UNORM = GL_RGB8,
        RGBA_8_BYTE_UNORM = GL_RGBA8,
        
        // 8-bit signed integer
        R_8_INT = GL_R8I,
        RG_8_INT = GL_RG8I,
        RGB_8_INT = GL_RGB8I,
        RGBA_8_INT = GL_RGBA8I,
        
        // 8-bit unsigned integer
        R_8_UINT = GL_R8UI,
        RG_8_UINT = GL_RG8UI,
        RGB_8_UINT = GL_RGB8UI,
        RGBA_8_UINT = GL_RGBA8UI,
        
        // 16-bit signed integer
        R_16_INT = GL_R16I,
        RG_16_INT = GL_RG16I,
        RGB_16_INT = GL_RGB16I,
        RGBA_16_INT = GL_RGBA16I,
        
        // 16-bit unsigned integer
        R_16_UINT = GL_R16UI,
        RG_16_UINT = GL_RG16UI,
        RGB_16_UINT = GL_RGB16UI,
        RGBA_16_UINT = GL_RGBA16UI,
        
        // 32-bit signed integer
        R_32_INT = GL_R32I,
        RG_32_INT = GL_RG32I,
        RGB_32_INT = GL_RGB32I,
        RGBA_32_INT = GL_RGBA32I,
        
        // 32-bit unsigned integer
        R_32_UINT = GL_R32UI,
        RG_32_UINT = GL_RG32UI,
        RGB_32_UINT = GL_RGB32UI,
        RGBA_32_UINT = GL_RGBA32UI,
                
        // 16-bit float
        R_16_FLOAT = GL_R16F,
        RG_16_FLOAT = GL_RG16F,
        RGB_16_FLOAT = GL_RGB16F,
        RGBA_16_FLOAT = GL_RGBA16F,
        
        // 32-bit float
        R_32_FLOAT = GL_R32F,
        RG_32_FLOAT = GL_RG32F,
        RGB_32_FLOAT = GL_RGB32F,
        RGBA_32_FLOAT = GL_RGBA32F,
        
        // depth
        DEPTH_COMPONENT_16 = GL_DEPTH_COMPONENT16,
		DEPTH_COMPONENT_24 = GL_DEPTH_COMPONENT24,
        DEPTH_COMPONENT_32 = GL_DEPTH_COMPONENT32,
		DEPTH_COMPONENT_32_FLOAT = GL_DEPTH_COMPONENT32F
        // TODO: depthstencil
	};

	static int getMaxTextureSize();
	static GLfloat getLargestSupportedAnisotropy();

	virtual ~GLTexture();

	// binds this texture object to the currently active texture unit	
	void bind();

	// unbinds a texture from the currently active texture unit
	void unbind();

	GLuint getTextureId();
	GLenum getTarget();
	GLTextureInternalFormat getInternalFormat();
	int getNumComponents();
	int getNumBitsPerComponent();

	GLTextureFilterMode getMinFilterMode();
	GLTextureFilterMode getMagFilterMode();

	void setFilterModeNearest();
	void setFilterModeLinear();

	void setFilterMode( GLTextureFilterMode minAndMagMode );
	void setFilterMode( GLTextureFilterMode minFilterMode, GLTextureFilterMode magFilterMode );
	
	void setAnisotropicFiltering( GLfloat anisotropy );

	// eParam: GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R
	// TODO: use an enum
	void setWrapMode( GLenum eParam, GLint iMode );

	// iMode = GL_CLAMP_TO_EDGE, etc
	// TODO: use an enum
	virtual void setAllWrapModes( GLint iMode ) = 0;

	virtual void getFloat1Data( float* afOutput, int level = 0 );
	virtual void getFloat3Data( float* afOutput, int level = 0 );
	virtual void getFloat4Data( float* afOutput, int level = 0 );

	virtual void getUnsignedByte1Data( uint8_t* aubOutput, int level = 0 );
	virtual void getUnsignedByte3Data( uint8_t* aubOutput, int level = 0 );
	virtual void getUnsignedByte4Data( uint8_t* aubOutput, int level = 0 );

	virtual void dumpToCSV( QString filename ) = 0;
	virtual void dumpToTXT( QString filename, GLint level = 0, GLenum format = GL_RGBA, GLenum type = GL_FLOAT ) = 0;

protected:
	
	GLTexture( GLenum eTarget, GLTextureInternalFormat internalFormat );

private:

	void getTexImage( GLint level, GLenum format, GLenum type, void* avOutput );

	GLenum m_eTarget;
	GLuint m_iTextureId;	
	GLTextureInternalFormat m_eInternalFormat;

	int m_nComponents;
	int m_nBitsPerComponent;
	
};

#endif
