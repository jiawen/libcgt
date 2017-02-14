#pragma once

#include "libcgt/core/vecmath/Box3f.h"
#include "libcgt/core/vecmath/Matrix4f.h"
#include "libcgt/core/vecmath/Vector4f.h"
#include "libcgt/GL/PlanarVertexBufferCalculator.h"
#include "libcgt/GL/GL_45/drawables/Drawable.h"

class SolidBox : public GLDrawable
{
public:

    SolidBox( const Box3f& box = Box3f( { 1, 1, 1 } ),
        const Matrix4f& worldFromBox = Matrix4f::identity(),
        const Vector4f& color = Vector4f( 1.0f ) );

    void updatePositions( const Box3f& box,
        const Matrix4f& worldFromBox = Matrix4f::identity() );

    // Update the entire color array to the same value.
    void updateColors( const Vector4f& color );

private:

    static PlanarVertexBufferCalculator calculator();
};
