#pragma once

#include "Drawable.h"

#include <vecmath/Matrix4f.h>
#include "libcgt/GL/PlanarVertexBufferCalculator.h"

class Axes : public GLDrawable
{
public:

    Axes( const Matrix4f& worldFromAxes = Matrix4f::identity(),
        float axisLength = 1.0f );

private:

    static PlanarVertexBufferCalculator calculator();
};
