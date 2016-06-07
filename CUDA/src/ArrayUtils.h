#pragma once

#include <vector>
#include <common/Array1DView.h>
#include <common/Array2DView.h>
#include <common/Array3DView.h>
#include <vector_types.h>

#include "DeviceArray1D.h"
#include "DeviceArray2D.h"
#include "DeviceArray3D.h"

namespace libcgt { namespace cuda { namespace arrayutils {

// Host functions (CUDA types).
bool saveTXT( Array1DView< const int3 > array, const char* filename );

bool saveTXT( Array2DView< const float2 > array, const char* filename );
bool saveTXT( Array2DView< const float4 > array, const char* filename );

bool saveTXT( Array2DView< const uchar4 > array, const char* filename );

bool saveTXT( Array3DView< const int2 > array, const char* filename );
bool saveTXT( Array3DView< const int3 > array, const char* filename );
bool saveTXT( Array3DView< const int4 > array, const char* filename );

// Device function.
bool saveTXT( const DeviceArray1D< int3 >& array, const char* filename );

bool saveTXT( const DeviceArray2D< float >& array, const char* filename );
bool saveTXT( const DeviceArray2D< float2 >& array, const char* filename );
bool saveTXT( const DeviceArray2D< float4 >& array, const char* filename );

bool saveTXT( const DeviceArray2D< uchar4 >& array, const char* filename );

bool saveTXT( const DeviceArray3D< int2 >& array, const char* filename );
bool saveTXT( const DeviceArray3D< int3 >& array, const char* filename );
bool saveTXT( const DeviceArray3D< int4 >& array, const char* filename );

};

} } // cuda, libcgt
