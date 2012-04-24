#include "VertexPosition4fNormal3fTexture2fBoneId4iWeight4f.h"

VertexPosition4fNormal3fTexture2fBoneId4iWeight4f::VertexPosition4fNormal3fTexture2fBoneId4iWeight4f()
{

}

VertexPosition4fNormal3fTexture2fBoneId4iWeight4f::VertexPosition4fNormal3fTexture2fBoneId4iWeight4f( const Vector4f& _position, const Vector3f& _normal, const Vector2f& _texture,
	const Vector4i& _boneIds, const Vector4f& _weights ) :

	position( _position ),
	normal( _normal ),
	texture( _texture ),
	boneIds( _boneIds ),
	weights( _weights )
{

}    

// static
int VertexPosition4fNormal3fTexture2fBoneId4iWeight4f::numElements()
{
	return 5;
}

// static
int VertexPosition4fNormal3fTexture2fBoneId4iWeight4f::sizeInBytes()
{
	return 13 * sizeof( float ) + 4 * sizeof( int );
}

// static
D3D11_INPUT_ELEMENT_DESC VertexPosition4fNormal3fTexture2fBoneId4iWeight4f::s_layout[] =
{
	{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 4 * sizeof( float ), D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 7 * sizeof( float ), D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "TEXCOORD", 1, DXGI_FORMAT_R32G32B32A32_SINT, 0, 9 * sizeof( float ), D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "TEXCOORD", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 9 * sizeof( float ) + 4 * sizeof( int ), D3D11_INPUT_PER_VERTEX_DATA, 0 }
};
