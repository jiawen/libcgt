#ifndef VERTEX_POSITION4F_NORMAL3F_TEXTURE2F_H
#define VERTEX_POSITION4F_NORMAL3F_TEXTURE2F_H

#include <D3D11.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

struct VertexPosition4fNormal3fTexture2f
{
	VertexPosition4fNormal3fTexture2f();
	VertexPosition4fNormal3fTexture2f( const Vector4f& _position, const Vector3f& _normal, const Vector2f& _texture );

	Vector4f position;
	Vector3f normal;
	Vector2f texture;

	static int numElements();
	static int sizeInBytes();
	static D3D11_INPUT_ELEMENT_DESC s_layout[];
};

#endif // VERTEX_POSITION4F_NORMAL3F_TEXTURE2F_H
