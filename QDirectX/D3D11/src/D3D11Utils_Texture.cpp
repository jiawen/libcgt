#include "D3D11Utils_Texture.h"

#include <common/ArrayUtils.h>

#include <color/ColorUtils.h>

#include "DynamicTexture2D.h"
#include "StagingTexture2D.h"

// static
std::shared_ptr< DynamicTexture2D > D3D11Utils_Texture::createTextureFromFile( ID3D11Device* pDevice, QString filename, bool flipUD )
{
	Image4ub im( filename );
	return createTextureFromImage( pDevice, im, flipUD );
}

// static
std::shared_ptr< DynamicTexture2D > D3D11Utils_Texture::createTextureFromImage( ID3D11Device* pDevice, const Image1f& im, bool flipUD )
{
	std::shared_ptr< DynamicTexture2D > pTexture( DynamicTexture2D::createFloat1( pDevice, im.width(), im.height() ) );
	copyImageToTexture( im, pTexture, flipUD );
	return pTexture;
}

// static
std::shared_ptr< DynamicTexture2D > D3D11Utils_Texture::createTextureFromImage( ID3D11Device* pDevice, const Image4f& im, bool flipUD )
{
	std::shared_ptr< DynamicTexture2D > pTexture( DynamicTexture2D::createFloat4( pDevice, im.width(), im.height() ) );
	copyImageToTexture( im, pTexture, flipUD );
	return pTexture;
}

// static
std::shared_ptr< DynamicTexture2D > D3D11Utils_Texture::createTextureFromImage( ID3D11Device* pDevice, const Image4ub& im, bool flipUD )
{
	std::shared_ptr< DynamicTexture2D > pTexture( DynamicTexture2D::createUnsignedByte4( pDevice, im.width(), im.height() ) );
	copyImageToTexture( im, pTexture, flipUD );
	return pTexture;
}


// static
void D3D11Utils_Texture::copyImageToTexture( const Image1f& im, std::shared_ptr< DynamicTexture2D > tex, bool flipUD )
{
	int width = im.width();
	int height = im.height();

	D3D11_MAPPED_SUBRESOURCE mapping = tex->mapForWriteDiscard();

	const float* sourceData = im.pixels();
	ubyte* destDataBytes = reinterpret_cast< ubyte* >( mapping.pData );

	// if the pitch matches and no flip is requested
	// then just directly copy
	if( mapping.RowPitch == width * sizeof( float ) && !flipUD )
	{
		float* destDataFloat = reinterpret_cast< float* >( mapping.pData );
		memcpy( destDataFloat, sourceData, width * height * sizeof( float ) );
	}
	// otherwise, have to go row by row
	for( int y = 0; y < height; ++y )
	{
		int yy = flipUD ? height - y - 1 : y;

		const float* sourceRow = &( sourceData[ yy * width ] );
		ubyte* destRow = &( destDataBytes[ y * mapping.RowPitch ] );

		memcpy( destRow, sourceRow, width * sizeof( float ) );
	}	

	tex->unmap();
}

// static
void D3D11Utils_Texture::copyImageToTexture( const Image4f& im, std::shared_ptr< DynamicTexture2D > tex, bool flipUD )
{
	int width = im.width();
	int height = im.height();

	D3D11_MAPPED_SUBRESOURCE mapping = tex->mapForWriteDiscard();

	const float* sourceData = im.pixels();
	ubyte* destDataBytes = reinterpret_cast< ubyte* >( mapping.pData );

	// if the pitch matches and no flip is requested
	// then just directly copy
	if( mapping.RowPitch == 4 * width * sizeof( float ) && !flipUD )
	{
		float* destDataFloat = reinterpret_cast< float* >( mapping.pData );
		memcpy( destDataFloat, sourceData, 4 * width * height * sizeof( float ) );
	}
	// otherwise, have to go row by row
	else
	{
		for( int y = 0; y < height; ++y )
		{
			int yy = flipUD ? height - y - 1 : y;

			const float* sourceRow = &( sourceData[ 4 * yy * width ] );
			ubyte* destRow = &( destDataBytes[ y * mapping.RowPitch ] );

			memcpy( destRow, sourceRow, 4 * width * sizeof( float ) );
		}
	}

	tex->unmap();
}


// static
void D3D11Utils_Texture::copyImageToTexture( const Image4ub& im, std::shared_ptr< DynamicTexture2D > tex, bool flipUD )
{
	int width = im.width();
	int height = im.height();

	D3D11_MAPPED_SUBRESOURCE mapping = tex->mapForWriteDiscard();

	const ubyte* sourceData = im.pixels();
	ubyte* destData = reinterpret_cast< ubyte* >( mapping.pData );

	// if the pitch matches and no flip is requested
	// then just directly copy
	if( mapping.RowPitch == 4 * width && !flipUD )
	{
		memcpy( destData, sourceData, 4 * width * height );
	}
	// otherwise, have to go row by row
	for( int y = 0; y < height; ++y )
	{
		int yy = flipUD ? height - y - 1 : y;

		const ubyte* sourceRow = &( sourceData[ 4 * yy * width ] );
		ubyte* destRow = &( destData[ y * mapping.RowPitch ] );

		memcpy( destRow, sourceRow, 4 * width * sizeof( ubyte ) );
	}	

	tex->unmap();
}

// static
void D3D11Utils_Texture::copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image1f& im )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;

	std::shared_ptr< StagingTexture2D > pST;

	if( desc.Format == DXGI_FORMAT_R32_FLOAT )
	{
		pST.reset( StagingTexture2D::createFloat1( pDevice, width, height ) );
	}

	if( desc.Format == DXGI_FORMAT_R32_FLOAT )
	{
		pST->copyFrom( pTexture );
		D3D11_MAPPED_SUBRESOURCE mapping = pST->mapForReadWrite();
		ubyte* sourceDataBytes = reinterpret_cast< ubyte* >( mapping.pData );

		if( mapping.RowPitch == width * sizeof( float ) )
		{
			memcpy( im.pixels(), sourceDataBytes, width * height * sizeof( float ) );
		}
		else
		{
			for( int y = 0; y < height; ++y )
			{
				float* sourceRow = reinterpret_cast< float* >( &( sourceDataBytes[ y * mapping.RowPitch ] ) );
				float* destinationRow = im.rowPointer( y );
				memcpy( destinationRow, sourceRow, width * sizeof( float ) );
			}
		}

		pST->unmap();
	}
	else
	{
		printf( "Warning: unable to copy texture to image, format is unsupported\n" );
	}
}

// static
void D3D11Utils_Texture::copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image1i& im )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;
	
	std::shared_ptr< StagingTexture2D > pST;

	if( desc.Format == DXGI_FORMAT_R16_UNORM )
	{
		pST.reset( StagingTexture2D::createUnsignedShort1UNorm( pDevice, width, height ) );
	}
	else if( desc.Format == DXGI_FORMAT_R16_UINT )
	{
		pST.reset( StagingTexture2D::createUnsignedShort1( pDevice, width, height ) );
	}

	if( desc.Format == DXGI_FORMAT_R16_UNORM ||
		desc.Format == DXGI_FORMAT_R16_UINT )
	{
		pST->copyFrom( pTexture );
		D3D11_MAPPED_SUBRESOURCE mapping = pST->mapForReadWrite();
		ubyte* sourceData = reinterpret_cast< ubyte* >( mapping.pData );

		for( int y = 0; y < height; ++y )
		{
			// TODO: memcpy to destination->rowPointer() on int
			// stuck with swizzling for shorts
			ushort* sourceRow = reinterpret_cast< ushort* >( &( sourceData[ y * mapping.RowPitch ] ) );
			for( int x = 0; x < width; ++x )
			{
				ushort r = sourceRow[ x ];
				im.setPixel( x, y, r );
			}
		}
		pST->unmap();
	}
	else
	{
		printf( "Warning: unable to copy texture to image, format is unsupported\n" );
	}
}

// static
void D3D11Utils_Texture::copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image4ub& im )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;

	std::shared_ptr< StagingTexture2D > pST;

	if( desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM )
	{
		pST.reset( StagingTexture2D::createUnsignedByte4( pDevice, width, height ) );
		pST->copyFrom( pTexture );
		D3D11_MAPPED_SUBRESOURCE mapping = pST->mapForReadWrite();
		ubyte* sourceData = reinterpret_cast< ubyte* >( mapping.pData );

		if( mapping.RowPitch == 4 * width )
		{
			memcpy( im.pixels(), sourceData, 4 * width * height );
		}
		else
		{
			for( int y = 0; y < height; ++y )
			{
				ubyte* sourceRow = &( sourceData[ y * mapping.RowPitch ] );
				ubyte* destinationRow = im.rowPointer( y );
				memcpy( destinationRow, sourceRow, 4 * width );
			}
		}
		pST->unmap();
	}
	else
	{
		printf( "Warning: unable to copy texture to image, format is unsupported\n" );
	}
}

// static
bool D3D11Utils_Texture::copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pSrc, Array2DView< ubyte4 > dstView )
{
	D3D11_TEXTURE2D_DESC desc;
	pSrc->GetDesc( &desc );

	if( desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM &&
		desc.Format != DXGI_FORMAT_R8G8B8A8_UINT )
	{
		return false;
	}

	if( desc.Width != dstView.width() ||
		desc.Height != dstView.height() )
	{
		return false;
	}

	// Copy into staging texture first
	std::unique_ptr< StagingTexture2D > pST( StagingTexture2D::createUnsignedByte4( pDevice, desc.Width, desc.Height ) );
	pST->copyFrom( pSrc );

	bool succeeded = copyTextureToImage( pST.get(), dstView );

	pST->unmap();

	return succeeded;
}

// static
bool D3D11Utils_Texture::copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pSrc, Array2DView< Vector4f > dstView )
{
	D3D11_TEXTURE2D_DESC desc;
	pSrc->GetDesc( &desc );

	if( desc.Format != DXGI_FORMAT_R32G32B32A32_FLOAT )
	{
		return false;
	}

	if( desc.Width != dstView.width() ||
		desc.Height != dstView.height() )
	{
		return false;
	}

	// Copy into staging texture first
	std::unique_ptr< StagingTexture2D > pST( StagingTexture2D::createFloat4( pDevice, desc.Width, desc.Height ) );
	pST->copyFrom( pSrc );

	bool succeeded = copyTextureToImage( pST.get(), dstView );

	pST->unmap();

	return succeeded;
}

// static
bool D3D11Utils_Texture::copyTextureToImage( StagingTexture2D* pSrc, Array2DView< ubyte4 > dstView )
{
	if( pSrc == nullptr )
	{
		return false;
	}

	if( pSrc->width() != dstView.width() ||
		pSrc->height() != dstView.height() )
	{
		return false;
	}

	D3D11_MAPPED_SUBRESOURCE srcMapping = pSrc->mapForReadWrite();

	Array2DView< ubyte4 > srcView
	(
		srcMapping.pData,
		pSrc->width(), pSrc->height(),
		4,
		srcMapping.RowPitch
	);

	bool succeeded = ArrayUtils::copy( srcView, dstView );

	pSrc->unmap();

	return succeeded;
}

// static
bool D3D11Utils_Texture::copyTextureToImage( StagingTexture2D* pSrc, Array2DView< Vector4f > dstView )
{
	if( pSrc == nullptr )
	{
		return false;
	}

	if( pSrc->width() != dstView.width() ||
		pSrc->height() != dstView.height() )
	{
		return false;
	}

	D3D11_MAPPED_SUBRESOURCE srcMapping = pSrc->mapForReadWrite();

	Array2DView< Vector4f > srcView
	(
		srcMapping.pData,
		pSrc->width(), pSrc->height(),
		16, // stride
		srcMapping.RowPitch
	);

	bool succeeded = ArrayUtils::copy( srcView, dstView );

	pSrc->unmap();

	return succeeded;
}

// static
void D3D11Utils_Texture::copyTextureToImage( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, Image4f& im )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;

	std::shared_ptr< StagingTexture2D > pST;

	if( desc.Format == DXGI_FORMAT_R32G32_FLOAT )
	{
		pST.reset( StagingTexture2D::createFloat2( pDevice, width, height ) );
	}
	else if( desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT )
	{
		pST.reset( StagingTexture2D::createFloat4( pDevice, width, height ) );
	}

	if( desc.Format == DXGI_FORMAT_R32G32_FLOAT )
	{
		pST->copyFrom( pTexture );
		D3D11_MAPPED_SUBRESOURCE mt = pST->mapForReadWrite();
		ubyte* sourceData = reinterpret_cast< ubyte* >( mt.pData );

		for( int y = 0; y < height; ++y )
		{
			float* sourceRow = reinterpret_cast< float* >( &( sourceData[ y * mt.RowPitch ] ) );
			for( int x = 0; x < width; ++x )
			{
				float r = sourceRow[ 2 * x ];
				float g = sourceRow[ 2 * x + 1 ];
				im.setPixel( x, y, Vector4f( r, g, 0, 1 ) );
			}
		}
		pST->unmap();
	}
	else if( desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT )
	{
		pST->copyFrom( pTexture );
		D3D11_MAPPED_SUBRESOURCE mapping = pST->mapForReadWrite();
		ubyte* sourceData = reinterpret_cast< ubyte* >( mapping.pData );

		if( mapping.RowPitch == 4 * width * sizeof( float ) )
		{
			memcpy( im.pixels(), sourceData, 4 * width * height * sizeof( float ) );
		}
		else
		{
			for( int y = 0; y < height; ++y )
			{
				float* sourceRow = reinterpret_cast< float* >( &( sourceData[ y * mapping.RowPitch ] ) );
				float* destinationRow = im.rowPointer( y );
				memcpy( destinationRow, sourceRow, 4 * width * sizeof( float ) );
			}
		}
		pST->unmap();
	}
	else
	{
		printf( "Warning: unable to copy texture to image, format is unsupported\n" );
	}
}

// static
void D3D11Utils_Texture::saveTextureToPFM( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;

	switch( desc.Format )
	{
		case DXGI_FORMAT_R32_FLOAT:
		{
			Image1f im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;
		}
		case DXGI_FORMAT_R32G32_FLOAT:
		case DXGI_FORMAT_R32G32B32A32_FLOAT:
		{
			Image4f im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;		
		}
		default:
		{			
			printf( "Unsupported format, TODO: USHORT_16 for depths\n" );
			break;
		}
	}
}

// static
void D3D11Utils_Texture::saveTextureToPFM4( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;

	switch( desc.Format )
	{
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		{
			Image4f im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;		
		}
	default:
		{			
			printf( "Unsupported format, TODO: USHORT_16 for depths\n" );
			break;
		}
	}
}

// static
void D3D11Utils_Texture::saveTextureToPNG( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename, bool scale, float factor )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;

	switch( desc.Format )
	{
	case DXGI_FORMAT_R16_UNORM:
	case DXGI_FORMAT_R16_UINT:
		{
			Image1i im( width, height );
			copyTextureToImage( pDevice, pTexture, im );

			if( scale )
			{
				int n = im.width() * im.height();
				int* pixels = im.pixels();				
				for( int i = 0; i < n; ++i )
				{
					int p = pixels[ i ];
					float f = factor * p + 0.5f;
					p = static_cast< qint32 >( f );
					pixels[ i ] = ColorUtils::saturate( p );
				}
			}

			im.savePNG( filename );
			break;
		}
	case DXGI_FORMAT_R8G8B8A8_UNORM:
		{
			Image4ub im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;
		}
	case DXGI_FORMAT_R32_FLOAT:
		{
			Image1f im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;
		}
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		{
			Image4f im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;
		}
	}
}

// static
void D3D11Utils_Texture::saveTextureToTXT( ID3D11Device* pDevice, ID3D11Texture2D* pTexture, QString filename )
{
	D3D11_TEXTURE2D_DESC desc;
	pTexture->GetDesc( &desc );

	int width = desc.Width;
	int height = desc.Height;

	switch( desc.Format )
	{
	case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
		{
			Image1i im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.saveTXT( filename );
			break;
		}
	case DXGI_FORMAT_R32_FLOAT:
		{
			Image1f im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;
		}
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		{
			Image4f im( width, height );
			copyTextureToImage( pDevice, pTexture, im );
			im.save( filename );
			break;
		}
	default:
		{
			printf( "saveTextureToTXT: texture format unsupported\n" );
		}
	}
}
