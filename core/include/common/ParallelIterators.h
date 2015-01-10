#pragma once

#ifdef WIN32

#include <ppl.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3i.h>
#include "ProgressReporter.h"

class ParallelIterators
{
public:	

	template< typename Function >
	static void for2D( const Vector2i& count, const Function& func );

	template< typename Function >
	static void for2D( const Vector2i& count, QString progressPrefix, const Function& func );

	template< typename Function >
	static void for2D( const Vector2i& first, const Vector2i& count, const Function& func );

	template< typename Function >
	static void for2D( const Vector2i& first, const Vector2i& count, const Vector2i& step, const Function& func );

	template< typename Function >
	static void for2D( const Vector2i& first, const Vector2i& count, const Vector2i& step, QString progressPrefix, const Function& func );

	template< typename Function >
	static void for3D( const Vector3i& count, const Function& func );

	template< typename Function >
	static void for3D( const Vector3i& count, QString progressPrefix, const Function& func );

	template< typename Function >
	static void for3D( const Vector3i& first, const Vector3i& count, const Function& func );

	template< typename Function >
	static void for3D( const Vector3i& first, const Vector3i& count, const Vector3i& step, const Function& func );

	template< typename Function >
	static void for3D( const Vector3i& first, const Vector3i& count, const Vector3i& step, QString progressPrefix, const Function& func );
};

// static
template< typename Function >
inline void ParallelIterators::for2D( const Vector2i& count, const Function& func )
{
    ParallelIterators::for2D( Vector2i{ 0, 0 }, count, Vector2i{ 1, 1 }, func );
}

// static
template< typename Function >
inline void ParallelIterators::for2D( const Vector2i& count, QString progressPrefix, const Function& func )
{
    ParallelIterators::for2D( Vector2i{ 0, 0 }, count, Vector2i{ 1, 1 }, progressPrefix, func );
}

// static
template< typename Function >
inline void ParallelIterators::for2D( const Vector2i& first, const Vector2i& count, const Function& func )
{
    ParallelIterators::for2D( first, count, Vector2i{ 1, 1 }, func );
}

// static
template< typename Function >
inline void ParallelIterators::for2D( const Vector2i& first, const Vector2i& count, const Vector2i& step, const Function& func )
{
	Concurrency::parallel_for
	(
		0, count.y, step.y,
		[&]( int j )
		{
			int y = first.y + j;
			for( int i = 0; i < count.x; i += step.x )
			{
				int x = first.x + i;

				func( x, y );
			}
		}
	);
}

// static
template< typename Function >
inline void ParallelIterators::for2D( const Vector2i& first, const Vector2i& count, const Vector2i& step, QString progressPrefix, const Function& func )
{
	ProgressReporter pr( progressPrefix, count.y / step.y );
	
	Concurrency::parallel_for
	(
		0, count.y, step.y,
		[&]( int j )
		{
			int y = first.y + j;
			for( int i = 0; i < count.x; i += step.x )
			{
				int x = first.x + i;

				func( x, y );
			}
			pr.notifyAndPrintProgressString();
		}
	);
}

// static
template< typename Function >
inline void ParallelIterators::for3D( const Vector3i& count, const Function& func )
{
	ParallelIterators::for3D( Vector3i{ 0, 0, 0 }, count, Vector3i{ 1, 1, 1 }, func );
}

// static
template< typename Function >
inline void ParallelIterators::for3D( const Vector3i& count, QString progressPrefix, const Function& func )
{	
	ParallelIterators::for3D( Vector3i{ 0, 0, 0 }, count, Vector3i{ 1, 1, 1 }, progressPrefix, func );
}

// static
template< typename Function >
inline void ParallelIterators::for3D( const Vector3i& first, const Vector3i& count, const Function& func )
{
	ParallelIterators::for3D( first, count, Vector3i{ 1, 1, 1 }, func );
}

// static
template< typename Function >
inline void ParallelIterators::for3D( const Vector3i& first, const Vector3i& count, const Vector3i& step, const Function& func )
{
	Concurrency::parallel_for
	(
		0, count.z, step.z,
		[&]( int k )
		{
			int z = first.z + k;
			for( int j = 0; j < count.y; j += step.y )
			{
				int y = first.y + j;
				for( int i = 0; i < count.x; i += step.x )
				{
					int x = first.x + i;

					func( x, y, z );
				}
			}
		}
	);
}

// static
template< typename Function >
inline void ParallelIterators::for3D( const Vector3i& first, const Vector3i& count, const Vector3i& step, QString progressPrefix, const Function& func )
{
	ProgressReporter pr( progressPrefix, count.z / step.z );

	Concurrency::parallel_for
	(
		0, count.z, step.z,
		[&]( int k )
		{
			int z = first.z + k;
			for( int j = 0; j < count.y; j += step.y )
			{
				int y = first.y + j;
				for( int i = 0; i < count.x; i += step.x )
				{
					int x = first.x + i;

					func( x, y, z );
				}
			}

			pr.notifyAndPrintProgressString();
		}
	);
}

#endif
