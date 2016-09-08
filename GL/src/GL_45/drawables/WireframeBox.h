#pragma once

#include "Drawable.h"

#include <vecmath/Box3f.h>
#include <vecmath/Matrix4f.h>
#include <vecmath/Vector4f.h>
#include <PlanarVertexBufferCalculator.h>

class WireframeBox : public GLDrawable
{
public:

    WireframeBox( const Box3f& box = Box3f( { 1, 1, 1 } ),
        const Vector4f& color = Vector4f( 1.0f ) );

    void updatePositions( const Box3f& box,
        const Matrix4f& worldFromBox = Matrix4f::identity() );

    // Update the entire color array to the same value.
    void updateColors( const Vector4f& color );

private:

    static PlanarVertexBufferCalculator calculator();
};
