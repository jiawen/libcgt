#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"

// TODO: can get rid of this file once it's pulled into a pure function.
// static
Vector3f Vector2f::cross( const Vector2f& v0, const Vector2f& v1 )
{
    return Vector3f
    (
        0,
        0,
        v0.x * v1.y - v0.y * v1.x
    );
}
