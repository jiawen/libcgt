#pragma once

#include <vector_types.h>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/vecmath/Vector4f.h"

#include "libcgt/cuda/DeviceArray1D.h"
#include "libcgt/cuda/DeviceArray2D.h"
#include "libcgt/cuda/DeviceArray3D.h"

namespace libcgt { namespace cuda {

// Host functions (CUDA types).
bool saveTXT( Array1DReadView< int3 > view, const std::string& filename );

bool saveTXT( Array2DReadView< float2 > view, const std::string& filename );
bool saveTXT( Array2DReadView< float4 > view, const std::string& filename );

bool saveTXT( Array2DReadView< uchar4 > view, const std::string& filename );

bool saveTXT( Array3DReadView< ushort2 > view, const std::string& filename );

bool saveTXT( Array3DReadView< int2 > view, const std::string& filename );
bool saveTXT( Array3DReadView< int3 > view, const std::string& filename );
bool saveTXT( Array3DReadView< int4 > view, const std::string& filename );

// Copy device array to host then save to disk.
template< typename T >
bool saveTXT( const DeviceArray1D< T >& src, const std::string& filename );

template< typename T >
bool saveTXT( const DeviceArray2D< T >& src, const std::string& filename );

template< typename T >
bool saveTXT( const DeviceArray3D< T >& src, const std::string& filename );

} } // cuda, libcgt

#include "libcgt/cuda/ArrayUtils.inl"
