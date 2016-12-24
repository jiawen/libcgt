#include <limits>
#include <utility>

namespace libcgt { namespace core { namespace math {

template< typename T >
std::pair< T, T > minMax( Array2DReadView< const T > src )
{
    // Extra parenthesization to get around annoying Win32 min/max nonsense.
    T mn = ( std::numeric_limits< T >::max )();
    T mx = ( std::numeric_limits< T >::min )();
    for( int y = 0; y < src.height(); ++y )
    {
        for( int x = 0; x < src.width(); ++x )
        {
            T z = src[ {x, y} ];
            if( z < mn )
            {
                mn = z;
            }
            if( z > mx )
            {
                mx = z;
            }
        }
    }

    return{ mn, mx };
}

} } } // math, core, libcgt
