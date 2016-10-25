#include "Vector2i.h"

#include "Vector3i.h"

// TODO: can get rid of this file once it's pulled into a pure function.
// static
Vector3i Vector2i::cross( const Vector2i& v0, const Vector2i& v1 )
{
    return
    {
        0,
        0,
        v0.x * v1.y - v0.y * v1.x
    };
}
