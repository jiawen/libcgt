#pragma once

#include <vector>
#include <common/Array2D.h>
#include <common/Array3D.h>
#include <vector_types.h>

#include "DeviceVector.h"
#include "DeviceArray2D.h"
#include "DeviceArray3D.h"

namespace libcgt
{
	namespace cuda
	{
		class ArrayUtils
		{
		public:
			
			// host
			static bool saveTXT( const Array2D< float2 >& array, const char* filename );
			static bool saveTXT( const Array2D< float4 >& array, const char* filename );

			static bool saveTXT( const std::vector< int3 >& array, const char* filename );

			static bool saveTXT( const Array3D< int2 >& array, const char* filename );
			static bool saveTXT( const Array3D< int3 >& array, const char* filename );
			static bool saveTXT( const Array3D< int4 >& array, const char* filename );

			// device
			static bool saveTXT( const DeviceArray2D< float >& array, const char* filename );
			static bool saveTXT( const DeviceArray2D< float4 >& array, const char* filename );

			//static bool saveTXT( const DeviceArray2D< uchar4 >& array, const char* filename );

			static bool saveTXT( const DeviceVector< int3 >& array, const char* filename );

			static bool saveTXT( const DeviceArray3D< int2 >& array, const char* filename );
			static bool saveTXT( const DeviceArray3D< int3 >& array, const char* filename );
			static bool saveTXT( const DeviceArray3D< int4 >& array, const char* filename );
		};

	}
}
