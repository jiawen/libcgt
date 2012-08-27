#pragma once

#include <common/Array2D.h>
#include <vector_types.h>

#include "DeviceArray2D.h"

namespace libcgt
{
	namespace cuda
	{
		class ArrayUtils
		{
		public:

			static bool saveTXT( const Array2D< float2 >& array, const char* filename );
			static bool saveTXT( const Array2D< float4 >& array, const char* filename );

			//static bool saveTXT( const DeviceArray2D< uchar4 >& array, const char* filename );
			static bool saveTXT( const DeviceArray2D< float >& array, const char* filename );
			static bool saveTXT( const DeviceArray2D< float4 >& array, const char* filename );
		};

	}
}
