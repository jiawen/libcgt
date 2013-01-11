#pragma once

#include <memory>
#include <d3d11.h>

#include <common/BasicTypes.h>
#include <common/Array2D.h>

#include <io/NumberedFilenameBuilder.h>

#include <vecmath/Vector4f.h>

class RenderTarget;
class DepthStencilTarget;
class StagingTexture2D;

class SequenceExporter
{
public:

	// extension can be either "png" or "pfm"
	SequenceExporter( ID3D11Device* pDevice, int width, int height, QString prefix, QString extension = "png", int startFrameIndex = 0 );
	virtual ~SequenceExporter();

	D3D11_VIEWPORT viewport();
	RenderTarget* renderTarget();
	DepthStencilTarget* depthStencilTarget();

	// saves the current render target, depth stencil target, and viewport
	// so frames can be saved out without destroying UI state
	void begin();

	// call before starting a new frame to notify the exporter
	// to clears its color and depth stencil targets
	void beginFrame();

	// call after rendering a new frame to notify the exporter
	// to save the frame to disk and increment the internal frame index
	void endFrame();

	// restores the render target, depth stencil target, and viewport for UI
	void end();

	// resets the frame index,
	// in case user wants to start somewhere else, or re-start and overwrite
	void setFrameIndex( int i );

private:

	int m_frameIndex;
	int m_width;
	int m_height;
	NumberedFilenameBuilder m_builder;
	QString m_extension;

	ID3D11Device* m_pDevice;
	ID3D11DeviceContext* m_pImmediateContext;

	std::unique_ptr< RenderTarget > m_pRT;
	std::unique_ptr< DepthStencilTarget > m_pDST;
	std::unique_ptr< StagingTexture2D > m_pStagingTexture;
	D3D11_VIEWPORT m_viewport;

	Array2D< ubyte4 > m_image4ub;
	Array2D< Vector4f > m_image4f;

	// saved for begin / end
	D3D11_VIEWPORT m_savedViewport;
	ID3D11RenderTargetView* m_pSavedRTV;
	ID3D11DepthStencilView* m_pSavedDSV;
};
