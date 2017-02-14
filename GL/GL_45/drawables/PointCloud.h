#pragma once

#include "libcgt/GL/PlanarVertexBufferCalculator.h"
#include "libcgt/GL/GL_45/drawables/Drawable.h"

class PointCloud : public GLDrawable
{
public:

    PointCloud( int nComponents, int nPoints );

private:

    static PlanarVertexBufferCalculator calculator( int nComponents,
        int nPoints );
};
