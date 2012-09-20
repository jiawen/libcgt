#include "imageproc/Image4ub.h"

#include <QFile>
#include <QTextStream>
#include <QImage>
#include <QString>

#include <math/Arithmetic.h>
#include <math/MathUtils.h>
#include <color/ColorUtils.h>

#include <imageproc/Image4f.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Image4ub::Image4ub() :

	m_width( 0 ),
	m_height( 0 )

{

}

Image4ub::Image4ub( QString filename ) :

	m_width( 0 ),
	m_height( 0 )

{
	load( filename );	
}

Image4ub::Image4ub( int width, int height, const Vector4i& fill ) :

	m_width( width ),
	m_height( height ),
	m_data( 4 * m_width, m_height )

{
	int nPixels = m_width * m_height;
	for( int i = 0; i < nPixels; ++i )
	{
		m_data[ 4 * i ] = ColorUtils::saturate( fill.x );
		m_data[ 4 * i + 1 ] = ColorUtils::saturate( fill.y );
		m_data[ 4 * i + 2 ] = ColorUtils::saturate( fill.z );
		m_data[ 4 * i + 3 ] = ColorUtils::saturate( fill.w );
	}
}

Image4ub::Image4ub( const Vector2i& size, const Vector4i& fill ) :

	m_width( size.x ),
	m_height( size.y ),
	m_data( 4 * m_width, m_height )

{
	int nPixels = m_width * m_height;
	for( int i = 0; i < nPixels; ++i )
	{
		m_data[ 4 * i ] = ColorUtils::saturate( fill.x );
		m_data[ 4 * i + 1 ] = ColorUtils::saturate( fill.y );
		m_data[ 4 * i + 2 ] = ColorUtils::saturate( fill.z );
		m_data[ 4 * i + 3 ] = ColorUtils::saturate( fill.w );
	}
}

Image4ub::Image4ub( const Image4ub& copy ) :

	m_width( copy.m_width ),
	m_height( copy.m_height ),
	m_data( copy.m_data )

{

}

Image4ub::Image4ub( Image4ub&& move )
{
	m_width = move.m_width;
	m_height = move.m_height;
	m_data = std::move( move.m_data );

	move.m_width = -1;
	move.m_height = -1;
}

Image4ub& Image4ub::operator = ( const Image4ub& copy )
{
	if( this != &copy )
	{
		m_width = copy.m_width;
		m_height = copy.m_height;
		m_data = copy.m_data;
	}
	return *this;
}

Image4ub& Image4ub::operator = ( Image4ub&& move )
{
	if( this != &move )
	{
		m_width = move.m_width;
		m_height = move.m_height;
		m_data = std::move( move.m_data );

		move.m_width = -1;
		move.m_height = -1;
	}
	return *this;
}

bool Image4ub::isNull() const
{
	return( m_width <= 0 || m_height <= 0 );
}

int Image4ub::width() const
{
	return m_width;
}

int Image4ub::height() const
{
	return m_height;
}

Vector2i Image4ub::size() const
{
	return Vector2i( m_width, m_height );
}

int Image4ub::numPixels() const
{
	return width() * height();
}

const ubyte* Image4ub::pixels() const
{
	return m_data;
}

ubyte* Image4ub::pixels()
{
	return m_data;
}

ubyte* Image4ub::rowPointer( int y )
{
	return m_data.rowPointer( y );
}

void Image4ub::fillChannel( int channel, ubyte value )
{
	int n = 4 * numPixels();
	for( int k = channel; k < n; k += 4 )
	{
		m_data[k] = value;
	}
}

Vector4i Image4ub::pixel( int x, int y ) const
{
	x = MathUtils::clampToRangeExclusive( x, 0, width() );
	y = MathUtils::clampToRangeExclusive( y, 0, height() );

	int index = 4 * ( y * m_width + x );

	return Vector4i
	(
		m_data[ index ],
		m_data[ index + 1 ],
		m_data[ index + 2 ],
		m_data[ index + 3 ]
	);
}

Vector4i Image4ub::pixel( const Vector2i& xy ) const
{
	return pixel( xy.x, xy.y );
}

void Image4ub::setPixel( int x, int y, const Vector4i& pixel )
{
	int index = 4 * ( y * m_width + x );

	m_data[ index ] = ColorUtils::saturate( pixel[ 0 ] );
	m_data[ index + 1 ] = ColorUtils::saturate( pixel[ 1 ] );
	m_data[ index + 2 ] = ColorUtils::saturate( pixel[ 2 ] );
	m_data[ index + 3 ] = ColorUtils::saturate( pixel[ 3 ] );
}

void Image4ub::setPixel( const Vector2i& xy, const Vector4i& pixel )
{
	setPixel( xy.x, xy.y, pixel );
}

Vector4i Image4ub::bilinearSample( float x, float y ) const
{
	x = x - 0.5f;
	y = y - 0.5f;

	// clamp to edge
	x = MathUtils::clampToRange( x, 0, m_width );
	y = MathUtils::clampToRange( y, 0, m_height );

	int x0 = MathUtils::clampToRangeExclusive( Arithmetic::floorToInt( x ), 0, m_width );
	int x1 = MathUtils::clampToRangeExclusive( x0 + 1, 0, m_width );
	int y0 = MathUtils::clampToRangeExclusive( Arithmetic::floorToInt( y ), 0, m_height );
	int y1 = MathUtils::clampToRangeExclusive( y0 + 1, 0, m_height );

	float xf = x - ( x0 + 0.5f );
	float yf = y - ( y0 + 0.5f );

	Vector4f v00 = ColorUtils::intToFloat( pixel( x0, y0 ) );
	Vector4f v01 = ColorUtils::intToFloat( pixel( x0, y1 ) );
	Vector4f v10 = ColorUtils::intToFloat( pixel( x1, y0 ) );
	Vector4f v11 = ColorUtils::intToFloat( pixel( x1, y1 ) );

	Vector4f v0 = MathUtils::lerp( v00, v01, yf ); // x = 0
	Vector4f v1 = MathUtils::lerp( v10, v11, yf ); // x = 1

	Vector4f vf = MathUtils::lerp( v0, v1, xf );
	return ColorUtils::floatToInt( vf );
}

Image4ub Image4ub::flipLR() const
{
	Image4ub output( m_width, m_height );

	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			int xx = m_width - x - 1;

			Vector4i p = pixel( xx, y );
			output.setPixel( x, y, p );
		}
	}

	return output;
}

Image4ub Image4ub::flipUD() const
{
	// TODO: do memcpy per row
	Image4ub output( m_width, m_height );

	for( int y = 0; y < m_height; ++y )
	{
		int yy = m_height - y - 1;
		for( int x = 0; x < m_width; ++x )
		{
			Vector4i p = pixel( x, yy );
			output.setPixel( x, y, p );
		}
	}

	return output;
}

QImage Image4ub::toQImage()
{
	QImage q( m_width, m_height, QImage::Format_ARGB32 );

	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			Vector4i pi = pixel( x, y );
			QRgb rgba = qRgba( pi.x, pi.y, pi.z, pi.w );
			q.setPixel( x, y, rgba );
		}
	}

	return q;
}

bool Image4ub::load( QString filename )
{
	QImage q( filename );
	if( q.isNull() )
	{
		return false;
	}

	m_width = q.width();
	m_height = q.height();
	m_data.resize( 4 * m_width, m_height );

	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			QRgb p = q.pixel( x, y );
			Vector4i vi( qRed( p ), qGreen( p ), qBlue( p ), qAlpha( p ) );
			setPixel( x, y, vi );
		}
	}
	return true;
}

bool Image4ub::save( QString filename )
{
	if( filename.endsWith( ".txt", Qt::CaseInsensitive ) )
	{
		return saveTXT( filename );
	}
	else if( filename.endsWith( ".png", Qt::CaseInsensitive ) )
	{
		return toQImage().save( filename, "PNG" );
	}
	return false;
}

bool Image4ub::saveTXT( QString filename )
{
	QFile outputFile( filename );

	// try to open the file in write only mode
	if( !( outputFile.open( QIODevice::WriteOnly ) ) )
	{
		return false;
	}

	QTextStream outputTextStream( &outputFile );
	outputTextStream.setCodec( "UTF-8" );

	outputTextStream << "float4 image: width = " << m_width << ", height = " << m_height << "\n";
	outputTextStream << "[index] (x,y_dx) ((x,y_gl)): r g b a\n";

	int k = 0;
	for( int y = 0; y < m_height; ++y )
	{
		int yy = m_height - y - 1;

		for( int x = 0; x < m_width; ++x )
		{
			ubyte r = m_data[ 4 * k ];
			ubyte g = m_data[ 4 * k + 1 ];
			ubyte b = m_data[ 4 * k + 2 ];
			ubyte a = m_data[ 4 * k + 3 ];

			outputTextStream << "[" << k << "] (" << x << "," << y << ") ((" << x << "," << yy << ")): "
				<< r << " " << g << " " << b << " " << a << "\n";

			++k;
		}
	}

	outputFile.close();
	return true;
}
