#include "cameras/PerspectiveCameraPath.h"

#include <math/Arithmetic.h>
#include <math/MathUtils.h>

PerspectiveCameraPath::PerspectiveCameraPath()
{

}

void PerspectiveCameraPath::addKeyframe( const PerspectiveCamera& camera )
{
	m_keyFrames.push_back( camera );
}

int PerspectiveCameraPath::numKeyFrames()
{
	return m_keyFrames.size();
}

void PerspectiveCameraPath::clear()
{
	m_keyFrames.clear();
}

void PerspectiveCameraPath::removeLastKeyframe()
{
	size_t n = m_keyFrames.size();
	if( n > 0 )
	{
		m_keyFrames.pop_back();
	}
}

PerspectiveCamera PerspectiveCameraPath::getCamera( float t )
{
	int nKeyFrames = m_keyFrames.size();

	// before the first one
	if( t < 0 )
	{
		return m_keyFrames[ 0 ];
	}
	// after the last one
	if( t >= nKeyFrames - 1 )
	{
		return m_keyFrames[ nKeyFrames - 1 ];
	}

	int p1Index = Arithmetic::floorToInt( t );
	float u = t - p1Index;

	int p0Index = MathUtils::clampToRangeInt( p1Index - 1, 0, nKeyFrames );
	int p2Index = MathUtils::clampToRangeInt( p1Index + 1, 0, nKeyFrames );
	int p3Index = MathUtils::clampToRangeInt( p1Index + 2, 0, nKeyFrames );

	PerspectiveCamera c0 = m_keyFrames[ p0Index ];
	PerspectiveCamera c1 = m_keyFrames[ p1Index ];
	PerspectiveCamera c2 = m_keyFrames[ p2Index ];
	PerspectiveCamera c3 = m_keyFrames[ p3Index ];

	return PerspectiveCamera::cubicInterpolate( c0, c1, c2, c3, u );
}

void PerspectiveCameraPath::load( const char* filename )
{
	m_keyFrames.clear();

	FILE* fp = fopen( filename, "rb" );

	int nFrames;
	fread( &nFrames, sizeof( int ), 1, fp );

	for( int i = 0; i < nFrames; ++i )
	{
		PerspectiveCamera c;
		fread( &c, sizeof( PerspectiveCamera ), 1, fp );
		m_keyFrames.push_back( c );
	}

	fclose( fp );
}

void PerspectiveCameraPath::save( const char* filename )
{
	FILE* fp = fopen( filename, "wb" );

	int nFrames = m_keyFrames.size();
	fwrite( &nFrames, sizeof( int ), 1, fp );

	for( int i = 0; i < nFrames; ++i )
	{
		PerspectiveCamera c = m_keyFrames[ i ];
		fwrite( &c, sizeof( PerspectiveCameraPath ), 1, fp );
	}

	fclose( fp );
}
