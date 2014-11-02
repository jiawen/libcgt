#pragma once

#include "common/Array2DView.h"
#include "math/Random.h"
#include "vecmath/Vector4f.h"

namespace libcgt
{
namespace core
{
namespace imageproc
{
namespace patterns
{
    // Explicitly instantiated for uint8_t, uint16_t, uint32_t, float, Vector2f, Vector3f, Vector4f.
    template< typename T >
    void createCheckerboard( Array2DView< T > image,
        int checkerSizeX, int checkerSizeY,
        const T& blackColor = T( 0 ), const T& whiteColor = T( 1 ) );    

    void createRandom( Array2DView< float > image, Random& random );
    void createRandom( Array2DView< Vector4f> image, Random& random );
}
}
}
}
