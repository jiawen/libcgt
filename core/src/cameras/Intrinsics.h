#pragma once

#include <vecmath/Matrix3f.h>
#include <vecmath/Vector2f.h>

namespace libcgt { namespace core { namespace cameras {

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

    Matrix3f asMatrix() const;
};

// Computes the same camera intrinsics if the image had its y axis flipped.
// Focal length and the x component of the principal point remains the same.
// The y component of the principal point is set to height - y.
Intrinsics flipY( const Intrinsics& intrinsics, float height );

} } } // cameras, core, libcgt
