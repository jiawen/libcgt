#include "cameras/Intrinsics.h"

namespace libcgt { namespace core { namespace cameras {

// static
Intrinsics Intrinsics::fromMatrix( const Matrix3f& k )
{
    return
    {
        { k( 0, 0 ), k( 1, 1 ) },
        { k( 0, 2 ), k( 1, 2 ) }
    };
}

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

Intrinsics flipY( const Intrinsics& intrinsics, float height )
{
    Intrinsics output = intrinsics;
    output.principalPoint.y = height - output.principalPoint.y;
    return output;
}

} } } // cameras, core, libcgt
