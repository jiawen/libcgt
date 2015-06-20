#pragma once

#include <common/BasicTypes.h>
#include "common/Iterators.h"
#include <geometry/BoundingBox3f.h>

#include "DynamicVertexBuffer.h"
#include "D3D11Utils.h"

class D3D11Utils_Box
{
public:

    static D3D11_BOX createRange( uint x, uint width );
    static D3D11_BOX createRect( uint x, uint y, uint width, uint height );
    static D3D11_BOX createBox( uint x, uint y, uint z, uint width, uint height, uint depth );

};
