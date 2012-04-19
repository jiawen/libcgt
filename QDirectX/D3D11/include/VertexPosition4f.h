#pragma once

#include <D3D11.h>
#include <vecmath/Vector4f.h>

struct VertexPosition4f
{
	VertexPosition4f();
	VertexPosition4f( float x, float y, float z, float w );
	VertexPosition4f( const Vector4f& _position );

	Vector4f position;

	static int numElements();
	static int sizeInBytes();

	// static void createVertexInputElementDescription( int semanticIndex, int slot );
	// static void createInstanceInputElementDescription( int semanticIndex, int slot, int rate = 1 );

	// "POSITION" in slot 0
	static const D3D11_INPUT_ELEMENT_DESC s_layout[];

	// Instancing layout:
	// Bound as TEXCOORD1, in slot 1
	// step rate of 1
	static const D3D11_INPUT_ELEMENT_DESC s_defaultInstanceLayout[];
};
