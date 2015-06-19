#include "common/Comparators.h"

#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"
#include "vecmath/Vector4i.h"

// static
bool Comparators::indexAndDistanceLess( const std::pair< int, float >& a, const std::pair< int, float >& b )
{
	return a.second < b.second;
}

// static
bool Comparators::indexAndDistanceGreater( const std::pair< int, float >& a, const std::pair< int, float >& b )
{
	return a.second > b.second;
}

// static
bool Comparators::vector2iLexigraphicLess( const Vector2i& a, const Vector2i& b )
{
	if( a.x < b.x )
	{
		return true;
	}
	else if( a.x > b.x )
	{
		return false;
	}
	else
	{
		return a.y < b.y;
	}
}

// static
bool Comparators::vector3iLexigraphicLess( const Vector3i& a, const Vector3i& b )
{
	if( a.x < b.x )
	{
		return true;
	}
	else if( a.x > b.x )
	{
		return false;
	}
	else
	{
		return vector2iLexigraphicLess( a.yz(), b.yz() );
	}
}

// static
bool Comparators::vector4iLexigraphicLess( const Vector4i& a, const Vector4i& b )
{
	if( a.x < b.x )
	{
		return true;
	}
	else if( a.x > b.x )
	{
		return false;
	}
	else
	{
		return vector3iLexigraphicLess( a.yzw(), b.yzw() );
	}
}
