#include "imageproc/Patterns.h"

#include <cstdint>

namespace libcgt { namespace core { namespace imageproc {

void createRandom( Array2DWriteView< float > image, Random& random )
{
    for( int i = 0; i < image.width() * image.height(); ++i )
    {
        image[ i ] = random.nextFloat();
    }
}

void createRandom( Array2DWriteView< Vector4f > image, Random& random )
{
    for( int i = 0; i < image.width() * image.height(); ++i )
    {
        image[ i ] = random.nextVector4f();
    }
}

} } } // imageproc, core, libcgt
