#pragma once

#include "Drawable.h"

#include <geometry/TriangleMesh.h>
#include <vecmath/Matrix4f.h>

// TODO(jiawen): add namespaces and rename this
// libcgt::GL::drawable::TriangleMesh?

// Simple mesh rendering from position and normals only, no texture mapping.
// TODO(jiawen): index buffering
class TriangleMeshDrawable : public GLDrawable
{
public:

    TriangleMeshDrawable( const TriangleMesh& mesh );

private:

    static PlanarVertexBufferCalculator calculator( const TriangleMesh& mesh );
};
