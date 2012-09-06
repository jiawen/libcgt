#pragma once

struct ID3D11Device;
class DynamicVertexBuffer;

struct VertexPosition4f;
struct VertexPosition4fTexture2f;

class D3D11RectangleUtils
{
public:

	// ===== full screen rectangles =====
	// 
	// positions are in clip coordinates (-1,-1) --> (1,1)
	// so the projection matrix should be identity (i.e., don't use one)
	//
	// if uv00AtTopLeft is false, the texture coordinates are flipped upside down ((0,1) at the top left corner)

	// Create a DynamicVertexBuffer< VertexPosition4fTexture2f > of 6 vertices (2 triangles)
	static DynamicVertexBuffer* createFullScreenRectangle( ID3D11Device* pDevice,
		bool uv00AtTopLeft = true, float z = 1, float w = 1 );

	// writes a fullscreen rectangle (6 vertices, 2 triangles) into buffer
	static void writeFullScreenRectangle( VertexPosition4fTexture2f* vertexArray,
		bool uv00AtTopLeft = true, float z = 1, float w = 1 );

	// ===== screen aligned rectangles =====
	//
	// positions are in camera coordinates:
	// (x,y) --> (x+width, y+height)
	// (x,y) is the bottom left corner (y-axis points up)
	// so the projection matrix should be (0,0) --> (viewportWidth, viewportHeight)
	//
	// if uv00AtTopLeft is true (default), the texture coordinates are flipped upside down ((0,1) at the top left corner)

	// Create a DynamicVertexBuffer< VertexPosition4fTexture2f > of 6 vertices (2 triangles)
	static DynamicVertexBuffer* createScreenAlignedRectangle( ID3D11Device* pDevice,
		float x, float y, float width, float height,
		bool uv00AtTopLeft = true, float z = 1, float w = 1 );

	// writes a screen aligned rectangle (6 vertices, 2 triangles) into buffer
	static void writeScreenAlignedRectangle( VertexPosition4f* vertexArray,
		float x, float y, float width, float height,
		float z = 1, float w = 1 );
	static void writeScreenAlignedRectangle( VertexPosition4fTexture2f* vertexArray,
		float x, float y, float width, float height,
		bool uv00AtTopLeft = true, float z = 1, float w = 1 );

	// writes the positions of a screen aligned rectangle into vertexArray
	template< typename T >
	static void writeScreenAlignedRectanglePositions( T* vertexArray,
		float x, float y, float width, float height,
		float z, float w );

	// writes the texture coordinates of a screen aligned rectangle into vertexArray
	template< typename T >
	static void writeScreenAlignedRectangleTextureCoordinates( T* vertexArray,
		bool uv00AtTopLeft );
};

template< typename T >
// static
void D3D11RectangleUtils::writeScreenAlignedRectanglePositions( T* vertexArray,
	float x, float y, float width, float height, float z, float w )
{
	vertexArray[ 0 ].position = Vector4f( x, y, z, w );
	vertexArray[ 1 ].position = Vector4f( x + width, y, z, w );
	vertexArray[ 2 ].position = Vector4f( x, y + height, z, w );

	vertexArray[ 3 ].position = Vector4f( x, y + height, z, w );
	vertexArray[ 4 ].position = Vector4f( x + width, y, z, w );
	vertexArray[ 5 ].position = Vector4f( x + width, y + height, z, w );
}

template< typename T >
// static
void D3D11RectangleUtils::writeScreenAlignedRectangleTextureCoordinates( T* vertexArray,
	bool uv00AtTopLeft )
{
	if( uv00AtTopLeft )
	{
		vertexArray[ 0 ].texture = Vector2f( 0, 1 );
		vertexArray[ 1 ].texture = Vector2f( 1, 1 );
		vertexArray[ 2 ].texture = Vector2f( 0, 0 );

		vertexArray[ 3 ].texture = Vector2f( 0, 0 );
		vertexArray[ 4 ].texture = Vector2f( 1, 1 );
		vertexArray[ 5 ].texture = Vector2f( 1, 0 );
	}
	else
	{
		vertexArray[ 0 ].texture = Vector2f( 0, 0 );
		vertexArray[ 1 ].texture = Vector2f( 1, 0 );
		vertexArray[ 2 ].texture = Vector2f( 0, 1 );

		vertexArray[ 3 ].texture = Vector2f( 0, 1 );
		vertexArray[ 4 ].texture = Vector2f( 1, 0 );
		vertexArray[ 5 ].texture = Vector2f( 1, 1 );
	}
}
