#pragma once

#include "Drawable.h"

#include <vecmath/Rect2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>
#include <PlanarVertexBufferCalculator.h>

class TexturedRectangle : public GLDrawable
{
public:

    TexturedRectangle(
        const Rect2f& positionRect = Rect2f{ { -1, -1 }, { 2, 2 } },
        const Rect2f& texCoordsRect = Rect2f{ { 1, 1 } } );

    void updatePositions( const Rect2f& positionRect );

    // Updates the vertex positions of this rectangle with the first 4 elements
    // of positions.
    void updatePositions( const std::vector< Vector3f >& positions );

    // TODO(jiawen): take Array1DView instead.
    // Updates the vertex positions of this rectangle with the first 4 elements
    // of positions.
    void updatePositions( const std::vector< Vector4f >& positions );

    void updateTexCoords( const Rect2f& texCoordsRect );

    // TODO(jiawen): update tex coords from an Array1DView.

private:

    static PlanarVertexBufferCalculator calculator();
};
