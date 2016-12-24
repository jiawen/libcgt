#pragma once

#include "libcgt/core/vecmath/Vector4f.h"
#include "libcgt/core/vecmath/Vector2f.h"

struct VertexPosition4fTexture2f
{
    Vector4f position;
    Vector2f texcoord;

    static const int s_numAttributes = 2;
    static const int s_numComponents[ 2 ];

    // Byte offsets for each attribute, in bytes.
    // The first is usually 0 but doesn't have to be!
    static const int s_relativeOffsets[ 2 ];

    // TODO: add a "enum type" for each attribute: UNSIGNED_BYTE, INT, FLOAT, etc
    // TODO: add a "bool normalized" for each attribute
    //   GL uses numComponents and a separate format: i.e. 4, float
    //   DX uses a single DXGI_FORMAT

    // TODO: strides, etc
    // static const D3D11_INPUT_ELEMENT_DESC s_layout[];
};
