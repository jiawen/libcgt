#pragma once

namespace libcgt
{
	namespace cuda
	{
		template< typename T >
		__inline__ __device__
		void swap( T& a, T& b )
		{
			T x = a;
			a = b;
			b = x;
		}
	}
}
