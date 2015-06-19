#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

namespace libcgt
{
	namespace cuda
	{
		class Rect2i
		{
		public:

			__inline__ __device__ __host__
			Rect2i();

			__inline__ __device__ __host__
			Rect2i( int left, int bottom, int width, int height );

			__inline__ __device__ __host__
			Rect2i( const int2& origin, const int2& size );

			__inline__ __device__ __host__
			int left() const;

			__inline__ __device__ __host__
			int right() const;

			__inline__ __device__ __host__
			int bottom() const;

			__inline__ __device__ __host__
			int top() const;

			__inline__ __device__ __host__
			int2 bottomLeft() const;

			__inline__ __device__ __host__
			int2 bottomRight() const;

			__inline__ __device__ __host__
			int2 topLeft() const;

			__inline__ __device__ __host__
			int2 topRight() const;

			__inline__ __device__ __host__
			int2 origin() const;

			__inline__ __device__ __host__
			int2 size() const;

			__inline__ __device__ __host__
			int area() const;

			__inline__ __device__ __host__
			Rect2i flippedUD( int height ) const;

		private:

			int2 m_origin;
			int2 m_size;
		};
	}
}

#include "Rect2i.inl"