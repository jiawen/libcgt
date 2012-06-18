#include "imageproc/Image1i.h"

#include <QFile>
#include <QTextStream>

#include <math/Arithmetic.h>
#include <math/MathUtils.h>
#include <color/ColorUtils.h>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

Image1i::Image1i() :

	m_width( 0 ),
	m_height( 0 )

{

}

Image1i::Image1i( int width, int height, int32 fill ) :

	m_width( width ),
	m_height( height ),
	m_data( width, height )

{
	m_data.fill( fill );
}

Image1i::Image1i( const Vector2i& size, int32 fill ) :

	m_width( size.x ),
	m_height( size.y ),
	m_data( size.x, size.y )

{
	m_data.fill( fill );
}

Image1i::Image1i( const Image1i& copy ) :

	m_width( copy.m_width ),
	m_height( copy.m_height ),
	m_data( copy.m_data )

{

}

bool Image1i::isNull() const
{
	return( m_width <= 0 || m_height <= 0 );
}

int Image1i::width() const
{
	return m_width;
}

int Image1i::height() const
{
	return m_height;
}

Vector2i Image1i::size() const
{
	return Vector2i( m_width, m_height );
}

const int32* Image1i::pixels() const
{
	return m_data;
}

int32* Image1i::pixels()
{
	return m_data;
}

int32* Image1i::rowPointer( int y )
{
	return m_data.rowPointer( y );
}

int32 Image1i::pixel( int x, int y ) const
{
	return m_data[ y * m_width + x ];
}

int32 Image1i::pixel( const Vector2i& xy ) const
{
	return pixel( xy.x, xy.y );
}

void Image1i::setPixel( int x, int y, int32 pixel )
{
	m_data[ y * m_width + x ] = pixel;
}

void Image1i::setPixel( const Vector2i& xy, int32 pixel )
{
	setPixel( xy.x, xy.y, pixel );
}

Image1i Image1i::flipUD()
{
	// TODO: do memcpy per row
	Image1i output( m_width, m_height );

	for( int y = 0; y < m_height; ++y )
	{
		int yy = m_height - y - 1;
		for( int x = 0; x < m_width; ++x )
		{
			int p = pixel( x, yy );
			output.setPixel( x, y, p );
		}
	}

	return output;
}

int32 Image1i::bilinearSample( float x, float y ) const
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

	float xf = x - x0;
	float yf = y - y0;

	float v00 = ColorUtils::intToFloat( pixel( x0, y0 ) );
	float v01 = ColorUtils::intToFloat( pixel( x0, y1 ) );
	float v10 = ColorUtils::intToFloat( pixel( x1, y0 ) );
	float v11 = ColorUtils::intToFloat( pixel( x1, y1 ) );

	float v0 = MathUtils::lerp( v00, v01, yf ); // x = 0
	float v1 = MathUtils::lerp( v10, v11, yf ); // x = 1

	float vf = MathUtils::lerp( v0, v1, xf );
	return static_cast< int32 >( ColorUtils::floatToInt( vf ) );
}

QImage Image1i::toQImage()
{
	QImage q( m_width, m_height, QImage::Format_ARGB32 );

	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			int32 pi = pixel( x, y );
			QRgb rgba = qRgba( pi, pi, pi, 255 );
			q.setPixel( x, y, rgba );
		}
	}

	return q;
}

void Image1i::savePNG( QString filename )
{
	toQImage().save( filename, "PNG" );
}

bool Image1i::saveTXT( QString filename )
{
	QFile outputFile( filename );

	// try to open the file in write only mode
	if( !( outputFile.open( QIODevice::WriteOnly ) ) )
	{
		return false;
	}

	QTextStream outputTextStream( &outputFile );
	outputTextStream.setCodec( "UTF-8" );

	outputTextStream << "int1 image: width = " << m_width << ", height = " << m_height << "\n";
	outputTextStream << "[index] (x,y_dx) ((x,y_gl)): r\n";

	int k = 0;
	for( int y = 0; y < m_height; ++y )
	{
		int yy = m_height - y - 1;

		for( int x = 0; x < m_width; ++x )
		{
			int r = m_data[ k ];
			outputTextStream << "[" << k << "] (" << x << "," << y << ") ((" << x << "," << yy << ")): " << r << "\n";

			++k;
		}
	}

	outputFile.close();
	return true;
}
