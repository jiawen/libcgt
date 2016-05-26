#pragma once

#include <vecmath/Matrix3f.h>
#include <vecmath/Vector2f.h>

// A simple class describing camera intrinsics.
// All values have units of pixels and thus, requires an image resolution to
// interpret as angles, and a pixel size to interpret as rays in a given
// Euclidean space.
class Intrinsics
{
public:

    Vector2f focalLength;
    Vector2f principalPoint;

    // TODO(jiawen): support skew.

    operator Matrix3f() const;
};
