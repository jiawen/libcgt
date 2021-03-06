#pragma once

#include "libcgt/core/geometry/TriangleMesh.h"
#include "libcgt/core/vecmath/Matrix4f.h"
#include "libcgt/GL/GL_45/drawables/Drawable.h"

// TODO(jiawen): add namespaces and rename this
// libcgt::GL::drawable::PositionNormalTriangleMesh?

// Simple mesh rendering from position and normals only, no texture mapping.
// TODO(jiawen): index buffering
class TriangleMeshDrawable : public GLDrawable
{
public:

    TriangleMeshDrawable( const TriangleMesh& mesh );

private:

    static PlanarVertexBufferCalculator calculator( const TriangleMesh& mesh );
};
