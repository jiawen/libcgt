#pragma once

#include <D3D11.h>

class DepthStencilTarget
{
public:

	// TODO: test for failure in texture/view creation, return NULL
	// TODO: add size(), resize()
	// TODO: get QD3D11Widget to use this

	static DepthStencilTarget* createDepthFloat24StencilUnsignedByte8( ID3D11Device* pDevice, int width, int height );	

	// untested!
	static DepthStencilTarget* createDepthFloat32( ID3D11Device* pDevice, int width, int height );	

	virtual ~DepthStencilTarget();
	
	int width();
	int height();

	ID3D11Texture2D* texture();
	ID3D11DepthStencilView* depthStencilView();

private:

	DepthStencilTarget( ID3D11Device* pDevice, int width, int height, ID3D11Texture2D* pTexture );

	int m_width;
	int m_height;

	ID3D11Texture2D* m_pTexture;
	ID3D11DepthStencilView* m_pDepthStencilView;	

};
