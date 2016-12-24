#pragma once

#include "Drawable.h"

#include "libcgt/GL/PlanarVertexBufferCalculator.h"

class PointCloud : public GLDrawable
{
public:

    PointCloud( int nComponents, int nPoints );

private:

    static PlanarVertexBufferCalculator calculator( int nComponents,
        int nPoints );
};
