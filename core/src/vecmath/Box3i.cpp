#include "vecmath/Box3i.h"

#include <QString>

#include "math/MathUtils.h"
#include "vecmath/Vector3f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Box3i::Box3i() :
	m_origin( 0 ),
	m_size( 0 )
{

}

Box3i::Box3i( const Vector3i& size ) :
	m_origin( { 0, 0, 0 } ),
	m_size( { size } )
{

}

Box3i::Box3i( const Vector3i& origin, const Vector3i& size ) :

	m_origin( origin ),
	m_size( size )

{

}

Box3i::Box3i( std::initializer_list< int > os )
{
    m_origin.x = *( os.begin() );
    m_origin.y = *( os.begin() + 1 );
    m_origin.z = *( os.begin() + 2 );
    m_size.x = *( os.begin() + 3 );
    m_size.y = *( os.begin() + 4 );
    m_size.z = *( os.begin() + 5 );
}

Vector3i Box3i::origin() const
{
	return m_origin;
}

Vector3i& Box3i::origin()
{
	return m_origin;
}

Vector3i Box3i::size() const
{
	return m_size;
}

Vector3i& Box3i::size()
{
	return m_size;
}

int Box3i::left() const
{
	return std::min( m_origin.x, m_origin.x + m_size.x );
}

int Box3i::right() const
{
	return left() + width();
}

int Box3i::bottom() const
{
	return std::min( m_origin.y, m_origin.y + m_size.y );
}

int Box3i::top() const
{
	return bottom() + height();
}

int Box3i::back() const
{
	return std::min( m_origin.z, m_origin.z + m_size.z );
}

int Box3i::front() const
{
    return back() + depth();
}

Vector3i Box3i::leftBottomBack() const
{
    return{ left(), bottom(), back() };
}

Vector3i Box3i::rightBottomBack() const
{
    return{ right(), bottom(), back() };
}

Vector3i Box3i::leftTopBack() const
{
    return{ left(), top(), back() };
}

Vector3i Box3i::rightTopBack() const
{
    return{ right(), top(), back() };
}

Vector3i Box3i::leftBottomFront() const
{
    return{ left(), bottom(), front() };
}

Vector3i Box3i::rightBottomFront() const
{
    return{ right(), bottom(), front() };
}

Vector3i Box3i::leftTopFront() const
{
    return{ left(), top(), front() };
}

Vector3i Box3i::rightTopFront() const
{
    return{ right(), top(), front() };
}

Vector3i Box3i::minimum() const
{
    return MathUtils::minimum( m_origin, m_origin + m_size );
}

Vector3i Box3i::maximum() const
{
    return MathUtils::maximum( m_origin, m_origin + m_size );
}

int Box3i::width() const
{
	return std::abs( m_size.x );
}

int Box3i::height() const
{
	return std::abs( m_size.y );
}

int Box3i::depth() const
{
	return std::abs( m_size.z );
}

int Box3i::volume() const
{
	return std::abs( m_size.x * m_size.y * m_size.z );
}

Vector3f Box3i::center() const
{
	return m_origin + 0.5f * m_size;
}

bool Box3i::isStandardized() const
{
	return( m_size.x >= 0 && m_size.y >= 0 && m_size.z >= 0 );
}

Box3i Box3i::standardized() const
{
	Vector3i origin;
	Vector3i size;

	if( m_size.x > 0 )
	{
		origin.x = m_origin.x;
		size.x = m_size.x;
	}
	else
	{
		origin.x = m_origin.x + m_size.x;
		size.x = -m_size.x;
	}

	if( m_size.y > 0 )
	{
		origin.y = m_origin.y;
		size.y = m_size.y;
	}
	else
	{
		origin.y = m_origin.y + m_size.y;
		size.y = -m_size.y;
	}

	if( m_size.z > 0 )
	{
		origin.z = m_origin.z;
		size.z = m_size.z;
	}
	else
	{
		origin.z = m_origin.z + m_size.z;
		size.z = -m_size.z;
	}

	return Box3i( origin, size );
}

QString Box3i::toString() const
{
	QString out;

	out.append( "Box3f:\n" );
	out.append( "\torigin: " );
	out.append( m_origin.toString() );
	out.append( "\n\tsize: " );
	out.append( m_size.toString() );

	return out;
}

bool Box3i::contains( const Vector3i& p )
{
	return
	(
		( p.x >= left() ) &&
		( p.x < right() ) &&
		( p.y >= bottom() ) &&
		( p.y < top() ) &&
		( p.z >= back() ) &&
		( p.z < front() )
	);
}

Box3i Box3i::flippedUD( int height ) const
{
	Vector3i origin;
	origin.x = m_origin.x;
	origin.y = height - top();
	origin.z = m_origin.z;

	return Box3i( origin, m_size );
}

Box3i Box3i::flippedBF( int depth ) const
{
	Vector3i origin;
	origin.x = m_origin.x;
	origin.y = m_origin.y;
	origin.z = depth - front();

	return Box3i( origin, m_size );
}

Box3i Box3i::flippedUDBF( int height, int depth ) const
{
	Vector3i origin;
	origin.x = m_origin.x;
	origin.y = height - top();
	origin.z = depth - front();

	return Box3i( origin, m_size );
}

// static
Box3i Box3i::united( const Box3i& r0, const Box3i& r1 )
{
	Vector3i unitedMin = MathUtils::minimum( r0.leftBottomBack(), r1.leftBottomBack() );
	Vector3i unitedMax = MathUtils::maximum( r0.rightTopFront(), r1.rightTopFront() );
	
	return Box3i( unitedMin, unitedMax - unitedMin );
}
