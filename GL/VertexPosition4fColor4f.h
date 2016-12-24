#pragma once

#include "libcgt/core/vecmath/Vector4f.h"

struct VertexPosition4fColor4f
{
    Vector4f position;
    Vector4f color;

    static const int s_numAttributes = 2;
    static const int s_numComponents[ 2 ];

    // Byte offsets for each attribute, in bytes.
    // The first is usually 0 but doesn't have to be!
    static const int s_relativeOffsets[ 2 ];

    // TODO: elementSizeBytes(), including padding

    // TODO: format / GL_FORMAT?: float, half, etc
    //   GL uses numComponents and a separate format: i.e. 4, float
    //   DX uses a single DXGI_FORMAT
    // TODO: bool normalized

    // TODO: strides, etc
    // static const D3D11_INPUT_ELEMENT_DESC s_layout[];
};
