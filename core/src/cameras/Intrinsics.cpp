#include "cameras/Intrinsics.h"

Matrix3f Intrinsics::asMatrix() const
{
    Matrix3f k;

    k( 0, 0 ) = focalLength.x;
    //k( 0, 1 ) = skew;
    k( 0, 2 ) = principalPoint.x;
    k( 1, 1 ) = focalLength.y;
    k( 1, 2 ) = principalPoint.y;
    k( 2, 2 ) = 1.0f;

    return k;
}
