#include "SequenceExporter.h"

#include <io/PNGIO.h>
#include <io/PortableFloatMapIO.h>

#include "RenderTarget.h"
#include "DepthStencilTarget.h"
#include "StagingTexture2D.h"
#include "D3D11Utils.h"
#include "D3D11Utils_Texture.h"

SequenceExporter::SequenceExporter( ID3D11Device* pDevice, int width, int height, QString prefix, QString extension, int startFrameIndex ) :

	m_pDevice( pDevice ),
	m_width( width ),
	m_height( height ),
	m_frameIndex( startFrameIndex ),
	m_extension( extension ),
	m_builder( prefix, QString( "." ) + extension )
{
	pDevice->AddRef();

	if( extension == "png" )
	{
		m_pRT = std::unique_ptr< RenderTarget >( RenderTarget::createUnsignedByte4( pDevice, width, height ) );
		m_pStagingTexture = std::unique_ptr< StagingTexture2D >( StagingTexture2D::createUnsignedByte4( pDevice, width, height ) );
		m_image4ub.resize( width, height );
	}
	else if( extension == "pfm" )
	{
		m_pRT = std::unique_ptr< RenderTarget >( RenderTarget::createFloat4( pDevice, width, height ) );
		m_pStagingTexture = std::unique_ptr< StagingTexture2D >( StagingTexture2D::createFloat4( pDevice, width, height ) );
		m_image4f.resize( width, height );
	}
	m_pDST = std::unique_ptr< DepthStencilTarget >( DepthStencilTarget::createDepthFloat24StencilUnsignedByte8( pDevice, width, height ) );

	m_viewport = D3D11Utils::createViewport( width, height );

	pDevice->GetImmediateContext( &m_pImmediateContext );
}

// virtual
SequenceExporter::~SequenceExporter()
{
	m_pImmediateContext->Release();
	m_pDevice->Release();
}

D3D11_VIEWPORT SequenceExporter::viewport()
{
	return m_viewport;
}

RenderTarget* SequenceExporter::renderTarget()
{
	return m_pRT.get();
}

DepthStencilTarget* SequenceExporter::depthStencilTarget()
{
	return m_pDST.get();
}

void SequenceExporter::begin()
{
	// save render targets
	m_pImmediateContext->OMGetRenderTargets( 1, &m_pSavedRTV, &m_pSavedDSV );

	// save viewport
	UINT nViewports = 1;
	m_pImmediateContext->RSGetViewports( &nViewports, &m_savedViewport );

	// set new render targets
	auto rtv = m_pRT->renderTargetView();
	auto dsv = m_pDST->depthStencilView();
	m_pImmediateContext->OMSetRenderTargets( 1, &rtv, dsv );

	// set new viewport
	m_pImmediateContext->RSSetViewports( 1, &m_viewport );

	m_pSavedRTV->Release();
	m_pSavedDSV->Release();
}

void SequenceExporter::beginFrame()
{
	auto rtv = m_pRT->renderTargetView();
	auto dsv = m_pDST->depthStencilView();
	m_pImmediateContext->ClearRenderTargetView( rtv, Vector4f() );
	m_pImmediateContext->ClearDepthStencilView( dsv, D3D11_CLEAR_DEPTH, 1.0f, 0 );
}

void SequenceExporter::endFrame()
{
	m_pStagingTexture->copyFrom( m_pRT->texture() );	

	if( m_extension == "png" )
	{
		D3D11Utils_Texture::copyTextureToImage
		(
			m_pStagingTexture.get(),
			m_image4ub
		);
	}
	else
	{
		D3D11Utils_Texture::copyTextureToImage
		(
			m_pStagingTexture.get(),
			m_image4f
		);	
	}
	
	QString filename = m_builder.filenameForNumber( m_frameIndex );
	printf( "Saving frame %d as %s\n", m_frameIndex, qPrintable( filename ) );

	if( m_extension == "png" )
	{
		PNGIO::writeRGBA( filename, m_image4ub );
	}
	else
	{
		Array2DView< Vector3f > rgbView
		(
			m_image4f.rowPointer( 0 ),
			m_image4f.size(),
			16, // stride
			m_image4f.rowPitchBytes()
		);
		PortableFloatMapIO::writeRGB( filename, rgbView );
	}

	++m_frameIndex;
}

void SequenceExporter::end()
{
	// restore viewport
	m_pImmediateContext->RSSetViewports( 1, &m_savedViewport );
	// restore render target
	m_pImmediateContext->OMSetRenderTargets( 1, &m_pSavedRTV, m_pSavedDSV );
	m_pSavedDSV->Release();
	m_pSavedRTV->Release();
}

void SequenceExporter::setFrameIndex( int i )
{
	m_frameIndex = i;
}