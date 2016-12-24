#pragma once

#include "libcgt/core/common/ArrayView.h"

namespace libcgt { namespace core { namespace math {

template< typename T >
std::pair< T, T > minMax( Array2DReadView< T > src );

} } } // math, core, libcgt

#include "ArrayOps.inl"
