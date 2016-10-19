#pragma once

#include "Drawable.h"

#include <vecmath/Rect2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>
#include <PlanarVertexBufferCalculator.h>

// The default textured rectangle has vertices from (-1, -1) to (1, 1) and
// texture coordinates from (0, 0) to (1, 1). Which means you can draw a full
// screen rectangle without any projection * modelview matrix: the vertices are
// already in clip space. To place it anywhere on the screen, just set the
// viewport.
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
