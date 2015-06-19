#include "imageproc/Patterns.h"

#include <cstdint>

void libcgt::core::imageproc::patterns::createRandom( Array2DView< float > image, Random& random )
{
    for( int i = 0; i < image.width() * image.height(); ++i )
    {
        image[ i ] = random.nextFloat();
    }
}

void libcgt::core::imageproc::patterns::createRandom( Array2DView< Vector4f > image, Random& random )
{
    for( int i = 0; i < image.width() * image.height(); ++i )
    {        
        image[ i ] = random.nextVector4f();
    }
}

