#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <cutil_math.h>

#include "Rect2i.h"
#include "MathUtils.h"

namespace libcgt
{
	namespace cuda
	{
		class Box3f
		{
		public:

			__inline__ __host__ __device__
			Box3f();

			__inline__ __host__ __device__
			Box3f( float originX, float originY, float originZ, float width, float height, float depth );

			__inline__ __host__ __device__
			Box3f( float width, float height, float depth );

			__inline__ __host__ __device__
			Box3f( const float3& origin, const float3& size );

			__inline__ __host__ __device__
			Box3f( const float3& size );

			__inline__ __host__ __device__
			float left() const;

			__inline__ __host__ __device__
			float right() const;

			__inline__ __host__ __device__
			float bottom() const;

			__inline__ __host__ __device__
			float top() const;

			__inline__ __host__ __device__
			float back() const;

			__inline__ __host__ __device__
			float front() const;

			__inline__ __host__ __device__
			float3 leftBottomBack() const;

			__inline__ __host__ __device__
			float3 rightTopFront() const;

			__inline__ __host__ __device__
			float3 center() const;

			__inline__ __host__ __device__
			void getCorners( float3 corners[8] ) const;

			__inline__ __host__ __device__
			static bool intersect( const Box3f& r0, const Box3f& r1 );

			__inline__ __host__ __device__
			static bool intersect( const Box3f& r0, const Box3f& r1, Box3f& intersection );

			float3 m_origin;
			float3 m_size;
		};
	}
}

#include "Box3f.inl"