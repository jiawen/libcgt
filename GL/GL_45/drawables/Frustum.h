#pragma once

#include "libcgt/core/cameras/PerspectiveCamera.h"
#include "libcgt/GL/PlanarVertexBufferCalculator.h"
#include "libcgt/GL/GL_45/drawables/Drawable.h"

class Frustum : public GLDrawable
{
public:

    Frustum( const PerspectiveCamera& camera = PerspectiveCamera(),
        const Vector4f& color = Vector4f( 1.0f ) );

    void updatePositions( const PerspectiveCamera& camera );

    void updateColor( const Vector4f& color );

private:

    static PlanarVertexBufferCalculator calculator();
};
