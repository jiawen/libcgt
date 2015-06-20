#pragma once

#include <D3D11.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Vector4i.h>

struct VertexPosition4fNormal3fTexture2fBoneId4iWeight4f
{
    VertexPosition4fNormal3fTexture2fBoneId4iWeight4f();
    VertexPosition4fNormal3fTexture2fBoneId4iWeight4f( const Vector4f& _position, const Vector3f& _normal, const Vector2f& _texture,
        const Vector4i& _boneIds, const Vector4f& _weights );

    Vector4f position;
    Vector3f normal;
    Vector2f texture;
    Vector4i boneIds;
    Vector4f weights;

    static int numElements();
    static int sizeInBytes();
    static D3D11_INPUT_ELEMENT_DESC s_layout[];
};
