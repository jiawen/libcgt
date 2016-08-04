#pragma once

namespace libcgt { namespace core { namespace math {

template< typename T >
std::pair< T, T > minMax( Array2DView< const T > src );

} } } // math, core, libcgt

#include "ArrayOps.inl"
