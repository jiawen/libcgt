#pragma once

#include <memory>
#include <D3D11.h>
#include <D3DX11.h>

#include <common/Array2DView.h>

#include <imageproc/Image1f.h>
#include <imageproc/Image1i.h>
#include <imageproc/Image4f.h>
#include <imageproc/Image4ub.h>

#include <vecmath/Vector4f.h>

class DynamicTexture2D;
class StagingTexture2D;

class D3D11Utils_Texture
{
public:

	// Loads a texture from a standard image file, 8 bits per color channel, 32 bits per pixel
	// By default, Image4ub loads images such that the top left has coordinates (0,0) and is at memory, location 0
	// set flipUD to true to flip it up/down
	static std::shared_ptr< DynamicTexture2D > createTextureFromFile( ID3D11Device* pDevice, QString filename, bool flipUD = false );

	// image (cpu) --> new texture (GPU)
	static std::shared_ptr< DynamicTexture2D > createTextureFromImage( ID3D11Device* pDevice, const Image1f& im, bool flipUD = false );
	static std::shared_ptr< DynamicTexture2D > createTextureFromImage( ID3D11Device* pDevice, const Image4f& im, bool flipUD = false );
	static std::shared_ptr< DynamicTexture2D > createTextureFromImage( ID3D11Device* pDevice, const Image4ub& im, bool flipUD = false );

	// image (cpu) --> texture (gpu)
	static void copyImageToTexture( const Image1f& im, std::shared_ptr< DynamicTexture2D > tex, bool flipUD = false );
	static void copyImageToTexture( const Image4f& im, std::shared_ptr< DynamicTexture2D > tex, bool flipUD = false );
	static void copyImageToTexture( const Image4ub& im, std::shared_ptr< DynamicTexture2D > tex, bool flipUD = false );

	// texture (gpu) --> image (cpu)
	static void copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image1f& im );
	static void copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image1i& im );
	static void copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image4ub& im );
	static void copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image4f& im );

	// texture (gpu) --> image (cpu)
	// src->format() must be DXGI_FORMAT_R8G8B8A8_UNORM
	// src->size() must match dst->size()
	static bool copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pSrc, Array2DView< ubyte4 > dstView );
	static bool copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pSrc, Array2DView< Vector4f > dstView );
	
	// TODO: templatize StagingTexture2D on the format
	// src->format() must be DXGI_FORMAT_R8G8B8A8_UNORM
	static bool copyTextureToImage( StagingTexture2D* pSrc, Array2DView< ubyte4 > dstView );
	// src->format() must be DXGI_FORMAT_R32G32B32A32_FLOAT
	static bool copyTextureToImage( StagingTexture2D* pSrc, Array2DView< Vector4f > dstView );

	// texture (gpu) --> file (disk)
	static void saveTextureToPFM( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename );
	static void saveTextureToPFM4( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename );
	static void saveTextureToPNG( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename, bool scale = false, float factor = 1.f );
	static void saveTextureToTXT( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename );
};
