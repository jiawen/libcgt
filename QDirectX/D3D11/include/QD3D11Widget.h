#ifndef QD3D11WIDGET_H
#define QD3D11WIDGET_H

#include <D3D11.h>
#include <d3dx11.h>

#include <QWidget>
#include <QStack>

class QD3D11Widget : public QWidget
{
public:

	QD3D11Widget( QWidget* parent = NULL );
	virtual ~QD3D11Widget();

	HRESULT initialize( int width, int height );

	ID3D11Device* device() const;
	ID3D11DeviceContext* immediateContext() const;

	void clearBackBuffer( float* rgba, float depth = 1 );

	void clearBackBufferColor( float* rgba );
	void clearBackBufferDepth( float depth = 1 );
	void clearBackBufferDepthStencil( float depth, UINT8 stencil );

	// set output buffers to the internal back buffer (color and depth-stencil)
	void restoreBackBuffer();

	ID3D11Texture2D* backBufferColor();
	ID3D11RenderTargetView* backBufferRenderTargetView();

	ID3D11Texture2D* backBufferDepthStencil();
	ID3D11DepthStencilView* backBufferDepthStencilView();

protected:

	virtual void initializeD3D();
	virtual void resizeD3D( int width, int height );
	virtual void paintD3D();

	IDXGISwapChain* m_pSwapChain;
	ID3D11Device* m_pDevice;
	ID3D11DeviceContext* m_pImmediateContext;

	ID3D11Texture2D* m_pBackBuffer;
	ID3D11RenderTargetView* m_pBackBufferRenderTargetView;

	ID3D11Texture2D* m_pDepthStencilBuffer;
	ID3D11DepthStencilView* m_pDepthStencilView;

private:

	virtual QPaintEngine* paintEngine() const;
	virtual void paintEvent( QPaintEvent* e );
	virtual void resizeEvent( QResizeEvent* e );

	bool m_bD3DInitialized;

	// initialization
	HRESULT createSwapChainAndDevice( int width, int height );
	HRESULT createBackBufferRenderTargetView();
	HRESULT createDepthStencilBuffers( int width, int height );

	// resizing
	HRESULT resizeSwapChain( int width, int height );
	HRESULT resizeDepthStencilBuffer( int width, int height );

};

#endif // QD3D11WIDGET_H
