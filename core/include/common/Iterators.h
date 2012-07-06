#pragma once

#include <vecmath/Vector2i.h>
#include <vecmath/Vector3i.h>
#include "ProgressReporter.h"

class Iterators
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
inline void Iterators::for2D( const Vector2i& count, const Function& func )
{
	Iterators::for2D( Vector2i( 0, 0 ), count, Vector2i( 1, 1 ), func );
}

// static
template< typename Function >
inline void Iterators::for2D( const Vector2i& count, QString progressPrefix, const Function& func )
{
	Iterators::for2D( Vector2i( 0, 0 ), count, Vector2i( 1, 1 ), progressPrefix, func );
}

// static
template< typename Function >
inline void Iterators::for2D( const Vector2i& first, const Vector2i& count, const Function& func )
{
	Iterators::for2D( first, count, Vector2i( 1, 1 ), func );
}

// static
template< typename Function >
inline void Iterators::for2D( const Vector2i& first, const Vector2i& count, const Vector2i& step, const Function& func )
{
	for( int j = 0; j < count.y; j += step.y )
	{
		int y = first.y + j;
		for( int i = 0; i < count.x; i += step.x )
		{
			int x = first.x + i;

			func( x, y );
		}
	}
}

// static
template< typename Function >
inline void Iterators::for2D( const Vector2i& first, const Vector2i& count, const Vector2i& step, QString progressPrefix, const Function& func )
{
	ProgressReporter pr( progressPrefix, count.y / step.y );

	for( int j = 0; j < count.y; j += step.y )
	{
		int y = first.y + j;
		for( int i = 0; i < count.x; i += step.x )
		{
			int x = first.x + i;

			func( x, y );
		}
		pr.notifyAndPrintProgressString();
	}
}

// static
template< typename Function >
inline void Iterators::for3D( const Vector3i& count, const Function& func )
{
	Iterators::for3D( Vector3i( 0, 0, 0 ), count, Vector3i( 1, 1, 1 ), func );
}

// static
template< typename Function >
inline void Iterators::for3D( const Vector3i& count, QString progressPrefix, const Function& func )
{	
	Iterators::for3D( Vector3i( 0, 0, 0 ), count, Vector3i( 1, 1, 1 ), progressPrefix, func );
}

// static
template< typename Function >
inline void Iterators::for3D( const Vector3i& first, const Vector3i& count, const Function& func )
{
	Iterators::for3D( first, count, Vector3i( 1, 1, 1 ), func );
}

// static
template< typename Function >
inline void Iterators::for3D( const Vector3i& first, const Vector3i& count, const Vector3i& step, const Function& func )
{
	for( int k = 0; k < count.z; k += step.z )
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
}

// static
template< typename Function >
inline void Iterators::for3D( const Vector3i& first, const Vector3i& count, const Vector3i& step, QString progressPrefix, const Function& func )
{
	ProgressReporter pr( progressPrefix, count.z / step.z );

	for( int k = 0; k < count.z; k += step.z )
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
}
