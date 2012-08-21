#pragma once

#include <D3D11.h>
#include <D3DX11.h>
#include <d3dx11effect.h>

#include <vector>
#include <DXGI.h>

#include <cameras/Camera.h>
#include <imageproc/Image1i.h>
#include <imageproc/Image1f.h>
#include <imageproc/Image4f.h>
#include <imageproc/Image4ub.h>

#include "DynamicVertexBuffer.h"
#include "StaticDataBuffer.h"
#include "StaticStructuredBuffer.h"
#include "VertexPosition4f.h"
#include "VertexPosition4fTexture2f.h"
#include "VertexPosition4fColor4f.h"
#include "VertexPosition4fNormal3fTexture2f.h"

class D3D11Utils
{
public:

	// Returns a std::vector of DXGI adapter on this machine
	// Be sure to release the pointers
	static std::vector< IDXGIAdapter* > getDXGIAdapters();

	static D3D11_VIEWPORT createViewport( int width, int height );
	static D3D11_VIEWPORT createViewport( const Vector2i& wh );
	static D3D11_VIEWPORT createViewport( int topLeftX, int topLeftY, int width, int height, float zMin = 0, float zMax = 1 );
	static D3D11_VIEWPORT createViewport( const Rect2f& rect, float zMin = 0, float zMax = 1 );

	// creates a unit box [0,1]^3
	static std::vector< VertexPosition4fNormal3fTexture2f > createBox( bool normalsPointOutward = true );

	// TODO: move these into its own class
	template< typename T >
	static ID3D11InputLayout* createInputLayout( ID3D11Device* pDevice, ID3DX11EffectPass* pPass )
	{
		ID3D11InputLayout* pInputLayout;

		D3DX11_PASS_DESC passDesc;
		pPass->GetDesc( &passDesc );
		HRESULT hr = pDevice->CreateInputLayout( T::s_layout, T::numElements(), passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &pInputLayout );
		if( SUCCEEDED( hr ) )
		{
			return pInputLayout;
		}
		else
		{
			return nullptr;
		}
	}
	
	template< typename TVertex, typename TInstance >
	static ID3D11InputLayout* createInstancedInputLayout( ID3D11Device* pDevice, ID3DX11EffectPass* pPass )
	{
		ID3D11InputLayout* pInputLayout;

		D3DX11_PASS_DESC passDesc;
		pPass->GetDesc( &passDesc );

		// merge the input layouts
		int nVertexElements = TVertex::numElements();
		int nInstanceElements = TInstance::numElements();

		std::vector< D3D11_INPUT_ELEMENT_DESC > layout( nVertexElements + nInstanceElements );
		for( int i = 0; i < nVertexElements; ++i )
		{
			layout[i] = TVertex::s_layout[i];
		}
		for( int j = 0; j < nInstanceElements; ++j )
		{
			layout[ nVertexElements + j ] = TInstance::s_defaultInstanceLayout[j];
		}

		HRESULT hr = pDevice->CreateInputLayout( layout.data(), layout.size(), passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &pInputLayout );
		if( SUCCEEDED( hr ) )
		{
			return pInputLayout;
		}
		else
		{
			return nullptr;
		}
	}

	// TODO: take in a PerspectiveCamera
	// TODO: make a version that takes in a color, one that doesn't
	// TODO: use frustumLines
	// Create a DynamicVertexBuffer< VertexPosition4fColor4f >
	// it has length 24 (12 lines, 2 * 12 vertices)
	static std::shared_ptr< DynamicVertexBuffer >createFrustum( ID3D11Device* pDevice,
		const Vector3f& eye, std::vector< Vector3f > frustumCorners,
		const Vector4f& color = Vector4f( 1, 1, 1, 1 ) );

	static void writeFrustum( const Vector3f& eye, const std::vector< Vector3f >& frustumCorners, VertexPosition4f* vertexArray );
	static void writeFrustum( const Vector3f& eye, const std::vector< Vector3f >& frustumCorners, const Vector4f& color, VertexPosition4fColor4f* vertexArray );

	// Create a DynamicVertexBuffer of 6 vertices
	// each vertex is a VertexPosition4fColor4f
	static DynamicVertexBuffer* createAxes( ID3D11Device* pDevice );

	// writes a set of axes into buffer
	static void writeAxes( VertexPosition4fColor4f* vertexArray );	

	// Create a DynamicVertexBuffer of 6 vertices (2 triangles) for a fullscreen quad
	// each vertex is a VertexPosition4fTexture2f
	// positions is set in clip coordinates clip coordinates (-1,-1) --> (1,1)
	//   so the projection matrix should be identity (i.e., don't use one)
	// if uv00AtTopLeft is true (default), the texture coordinates are flipped upside down ((0,0) at the top left corner)
	static std::shared_ptr< DynamicVertexBuffer > createFullScreenQuad( ID3D11Device* pDevice, bool uv00AtTopLeft = true );

	// Create a DynamicVertexBuffer of 6 vertices (2 triangles) for a screen aligned quad
	// each vertex is a VertexPosition4fTexture2f with
	// position is set to (x,y) --> (x+width, y+height), z = 0, w = 1, (x,y) is the bottom left corner (y-axis points up)
	// if uv00AtTopLeft is true (default), the texture coordinates are flipped upside down ((0,0) at the top left corner)
	static std::shared_ptr< DynamicVertexBuffer > createScreenAlignedQuad( float x, float y, float width, float height, ID3D11Device* pDevice,
		bool uv00AtTopLeft = true );

	// writes a fullscreen quadrilateral (6 vertices, 2 triangles) into buffer
	// each vertex is a VertexPosition4fTexture2f
	// positions is set in clip coordinates clip coordinates (-1,-1) --> (1,1)
	//   so the projection matrix should be identity (i.e., don't use one)
	// if uv00AtTopLeft is true (default), the texture coordinates are flipped upside down ((0,0) at the top left corner)
	static void writeFullScreenQuad( VertexPosition4fTexture2f* vertexArray, bool uv00AtTopLeft = true );

	// writes a screen aligned quadrilateral (6 vertices, 2 triangles) into buffer
	// position is set to (x,y) --> (x+width, y+height), z = 0, w = 1, (x,y) is the bottom left corner (y-axis points up)	
	static void writeScreenAlignedQuad( float x, float y, float width, float height, VertexPosition4f* vertexArray );

	template< typename T >
	static void writeAxisAlignedQuad( float x, float y, float width, float height, T* vertexArray )
	{
		vertexArray[ 0 ].position = Vector4f( x, y, 0, 1 );
		vertexArray[ 1 ].position = Vector4f( x + width, y, 0, 1 );
		vertexArray[ 2 ].position = Vector4f( x, y + height, 0, 1 );

		vertexArray[ 3 ].position = vertexArray[ 2 ].position;
		vertexArray[ 4 ].position = vertexArray[ 1 ].position;
		vertexArray[ 5 ].position = Vector4f( x + width, y + height, 0, 1 );
	}

	// writes a screen aligned quadrilateral (6 vertices, 2 triangles) into buffer
	// position is set to (x,y) --> (x+width, y+height), z = 0, w = 1, (x,y) is the bottom left corner (y-axis points up)
	// if uv00AtTopLeft is true (default), the texture coordinates are flipped upside down ((0,0) at the top left corner)
	static void writeScreenAlignedQuad( float x, float y, float width, float height, VertexPosition4fTexture2f* vertexArray, bool uv00AtTopLeft = true );

	static bool saveFloat2BufferToTXT( ID3D11Device* pDevice, std::shared_ptr< StaticDataBuffer > pBuffer, QString filename );
	static bool saveFloat2BufferToTXT( ID3D11Device* pDevice, std::shared_ptr< StaticStructuredBuffer > pBuffer, QString filename );	

	// HACK
	template< typename T >
	static void saveSTLVectorToBinary( const std::vector< T >& input, const char* filename )
	{
		FILE* fp = fopen( filename, "wb" );
		uint size = static_cast< uint >( input.size() );
		fwrite( &size, sizeof( uint ), 1, fp );
		fwrite( &( input[ 0 ] ), sizeof( T ), size, fp );
		fflush( fp );
		fclose( fp );
	}

};
