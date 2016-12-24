#pragma once

#include <vector>
#include <vector_types.h>

#include <common/ArrayView.h>
#include <vecmath/Vector4f.h>

#include "libcgt/cuda/DeviceArray1D.h"
#include "libcgt/cuda/DeviceArray2D.h"
#include "libcgt/cuda/DeviceArray3D.h"

namespace libcgt { namespace cuda {

// Host functions (CUDA types).
bool saveTXT( Array1DReadView< int3 > array, const char* filename );

bool saveTXT( Array2DReadView< float2 > array, const char* filename );
bool saveTXT( Array2DReadView< float4 > array, const char* filename );

bool saveTXT( Array2DReadView< uchar4 > array, const char* filename );

bool saveTXT( Array3DReadView< ushort2 > array, const char* filename );

bool saveTXT( Array3DReadView< int2 > array, const char* filename );
bool saveTXT( Array3DReadView< int3 > array, const char* filename );
bool saveTXT( Array3DReadView< int4 > array, const char* filename );

// Device functions (CUDA types).
bool saveTXT( const DeviceArray1D< int3 >& array, const char* filename );

bool saveTXT( const DeviceArray2D< float >& array, const char* filename );
bool saveTXT( const DeviceArray2D< float2 >& array, const char* filename );
bool saveTXT( const DeviceArray2D< float4 >& array, const char* filename );

bool saveTXT( const DeviceArray2D< uchar4 >& array, const char* filename );

bool saveTXT( const DeviceArray3D< ushort2 >& array, const char* filename );

bool saveTXT( const DeviceArray3D< int2 >& array, const char* filename );
bool saveTXT( const DeviceArray3D< int3 >& array, const char* filename );
bool saveTXT( const DeviceArray3D< int4 >& array, const char* filename );

} } // cuda, libcgt
