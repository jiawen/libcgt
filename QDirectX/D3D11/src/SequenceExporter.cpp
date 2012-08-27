#include "SequenceExporter.h"

#include "RenderTarget.h"
#include "DepthStencilTarget.h"
#include "StagingTexture2D.h"
#include "D3D11Utils.h"

SequenceExporter::SequenceExporter( ID3D11Device* pDevice, int width, int height, QString prefix, QString extension, int startFrameIndex ) :

	m_width( width ),
	m_height( height ),
	m_frameIndex( startFrameIndex ),
	m_extension( extension ),
	m_builder( prefix, QString( "." ) + extension )
{

	if( extension == "png" )
	{
		m_pRT.reset( RenderTarget::createUnsignedByte4( pDevice, width, height ) );
		m_pStagingTexture.reset( StagingTexture2D::createUnsignedByte4( pDevice, width, height ) );
		m_image4ub = Image4ub( width, height );
	}
	else if( extension == "pfm" )
	{
		m_pRT.reset( RenderTarget::createFloat4( pDevice, width, height ) );
		m_pStagingTexture.reset( StagingTexture2D::createFloat4( pDevice, width, height ) );
		m_image4f = Image4f( width, height );
	}
	m_pDST.reset( DepthStencilTarget::createDepthFloat24StencilUnsignedByte8( pDevice, width, height ) );

	m_viewport = D3D11Utils::createViewport( width, height );

	pDevice->GetImmediateContext( &m_pImmediateContext );
}

// virtual
SequenceExporter::~SequenceExporter()
{
	m_pImmediateContext->Release();
}

D3D11_VIEWPORT SequenceExporter::viewport()
{
	return m_viewport;
}

std::shared_ptr< RenderTarget > SequenceExporter::renderTarget()
{
	return m_pRT;
}

std::shared_ptr< DepthStencilTarget > SequenceExporter::depthStencilTarget()
{
	return m_pDST;
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
		
	// TODO: D3D11Utils_Texture should have this
	// or StagingTexture should take this
	D3D11_MAPPED_SUBRESOURCE mt = m_pStagingTexture->mapForReadWrite();

	// HACK
	if( m_extension == "png" )
	{
		ubyte* sourceData = reinterpret_cast< ubyte* >( mt.pData );

		for( int y = 0; y < m_height; ++y )
		{
			ubyte* srcRow = &( sourceData[ y * mt.RowPitch ] );
			ubyte* dstRow = m_image4ub.rowPointer( y );
			memcpy( dstRow, srcRow, 4 * m_width );
		}
	}
	else
	{
		ubyte* sourceData = reinterpret_cast< ubyte* >( mt.pData );

		for( int y = 0; y < m_height; ++y )
		{
			float* srcRow = reinterpret_cast< float* >( &( sourceData[ y * mt.RowPitch ] ) );
			float* dstRow = m_image4f.rowPointer( y );
			memcpy( dstRow, srcRow, 4 * m_width * sizeof( float ) );
		}
	}

	m_pStagingTexture->unmap();


	QString filename = m_builder.filenameForNumber( m_frameIndex );
	printf( "Saving frame %d as %s\n", m_frameIndex, qPrintable( filename ) );

	if( m_extension == "png" )
	{
		m_image4ub.save( filename );
	}
	else
	{
		m_image4f.save( filename );
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