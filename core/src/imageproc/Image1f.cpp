#include "imageproc/Image1f.h"

#include <QFile>
#include <QDataStream>
#include <QTextStream>

#include "color/ColorUtils.h"
#include "math/Arithmetic.h"
#include "math/MathUtils.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector4i.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
Reference< Image1f > Image1f::loadPFM( QString filename )
{
	QFile inputFile( filename );

	// try to open the file in read only mode
	if( !( inputFile.open( QIODevice::ReadOnly ) ) )
	{		
		return NULL;
	}

	QTextStream inputTextStream( &inputFile );
	inputTextStream.setCodec( "ISO-8859-1" );

	// read header
	QString qsType;
	QString qsWidth;
	QString qsHeight;
	QString qsScale;	

	int width;
	int height;
	float scale;

	inputTextStream >> qsType;
	if( qsType != "Pf" )
	{
		inputFile.close();
		return NULL;
	}

	inputTextStream >> qsWidth >> qsHeight >> qsScale;

	width = qsWidth.toInt();
	height = qsHeight.toInt();
	scale = qsScale.toFloat();

	if( width < 0 || height < 0 || scale >= 0 )
	{
		inputFile.close();
		return NULL;
	}

	// close the text stream
	inputTextStream.setDevice( NULL );
	inputFile.close();

	// now reopen it again in binary mode
	if( !( inputFile.open( QIODevice::ReadOnly ) ) )
	{
		return NULL;
	}

	int headerLength = qsType.length() + qsWidth.length() + qsHeight.length() + qsScale.length() + 4;	

	QDataStream inputDataStream( &inputFile );
	inputDataStream.skipRawData( headerLength );

	Reference< Image1f > image = new Image1f( width, height );
	float buffer;

	for( int y = 0; y < height; ++y )
	{
		for( int x = 0; x < width; ++x )
		{
			int yy = height - y - 1;

			inputDataStream.readRawData( reinterpret_cast< char* >( &buffer ), sizeof( float ) );
			image->setPixel( x, yy, buffer );
		}
	}

	inputFile.close();
	return image;
}

Image1f::Image1f() :

m_width( 0 ),
m_height( 0 ),
m_data( NULL )

{

}

Image1f::Image1f( int width, int height, float fill ) :

m_width( width ),
m_height( height ),
m_data( m_width * m_height, fill )

{

}

Image1f::Image1f( const Vector2i& size, float fill ) :

m_width( size.x() ),
m_height( size.y() ),
m_data( m_width * m_height, fill )

{

}

Image1f::Image1f( const Image1f& copy ) :

m_width( copy.m_width ),
m_height( copy.m_height ),
m_data( copy.m_data.copy() )

{

}

Image1f::Image1f( Reference< Image1f > copy ) :

m_width( copy->m_width ),
m_height( copy->m_height ),
m_data( copy->m_data.copy() )

{

}

bool Image1f::isNull() const
{
	return( m_width <= 0 || m_height <= 0 );
}

int Image1f::width() const
{
	return m_width;
}

int Image1f::height() const
{
	return m_height;
}

Vector2i Image1f::size() const
{
	return Vector2i( m_width, m_height );
}

FloatArray Image1f::pixels()
{
	return m_data;
}

float Image1f::pixel( int x, int y ) const
{
	int index = y * m_width + x;

	return m_data[ index ];
}

float Image1f::pixel( const Vector2i& xy ) const
{
	return pixel( xy.x(), xy.y() );
}

void Image1f::setPixel( int x, int y, float pixel )
{
	int index = y * m_width + x;
	m_data[ index ] = pixel;
}

void Image1f::setPixel( const Vector2i& xy, float pixel )
{
	setPixel( xy.x(), xy.y(), pixel );
}

void Image1f::setPixel( int x, int y, int pixel )
{
	float f = ColorUtils::intToFloat( pixel );
	setPixel( x, y, f );
}

void Image1f::setPixel( const Vector2i& xy, int pixel )
{
	setPixel( xy.x(), xy.y(), pixel );
}

Reference< Image1f > Image1f::flipUD()
{
	Reference< Image1f > output = new Image1f( m_width, m_height );

	for( int y = 0; y < m_height; ++y )
	{
		int yy = m_height - y - 1;
		for( int x = 0; x < m_width; ++x )
		{
			float p = pixel( x, yy );
			output->setPixel( x, y, p );
		}
	}

	return output;
}

float Image1f::bilinearSample( float x, float y ) const
{
	x = x - 0.5f;
	y = y - 0.5f;

	// clamp to edge
	x = MathUtils::clampToRangeFloat( x, 0, m_width );
	y = MathUtils::clampToRangeFloat( y, 0, m_height );

	int x0 = MathUtils::clampToRangeInt( Arithmetic::floorToInt( x ), 0, m_width );
	int x1 = MathUtils::clampToRangeInt( x0 + 1, 0, m_width );
	int y0 = MathUtils::clampToRangeInt( Arithmetic::floorToInt( y ), 0, m_height );
	int y1 = MathUtils::clampToRangeInt( y0 + 1, 0, m_height );

	float xf = x - x0;
	float yf = y - y0;

	float v00 = pixel( x0, y0 );
	float v01 = pixel( x0, y1 );
	float v10 = pixel( x1, y0 );
	float v11 = pixel( x1, y1 );

	float v0 = MathUtils::lerp( v00, v01, yf ); // x = 0
	float v1 = MathUtils::lerp( v10, v11, yf ); // x = 1

	return MathUtils::lerp( v0, v1, xf );
}

float Image1f::bilinearSample( const Vector2f& xy ) const
{
	return bilinearSample( xy.x(), xy.y() );
}

QImage Image1f::toQImage()
{
	QImage q( m_width, m_height, QImage::Format_ARGB32 );

	for( int y = 0; y < m_height; ++y )
	{
		for( int x = 0; x < m_width; ++x )
		{
			float pf = pixel( x, y );
			int pi = ColorUtils::floatToInt( pf );
			QRgb rgba = qRgba( pi, pi, pi, 255 );
			q.setPixel( x, m_height - y - 1, rgba );
		}
	}

	return q;
}

bool Image1f::savePNG( QString filename )
{
	return toQImage().save( filename, "PNG" );
}

bool Image1f::saveTXT( QString filename )
{
	QFile outputFile( filename );

	// try to open the file in write only mode
	if( !( outputFile.open( QIODevice::WriteOnly ) ) )
	{
		return false;
	}

	QTextStream outputTextStream( &outputFile );
	outputTextStream.setCodec( "UTF-8" );

	outputTextStream << "float1 image: width = " << m_width << ", height = " << m_height << "\n";
	outputTextStream << "[index] (x,y_dx) ((x,y_gl)): r\n";

	int k = 0;
	for( int y = 0; y < m_height; ++y )
	{
		int yy = m_height - y - 1;

		for( int x = 0; x < m_width; ++x )
		{
			float r = m_data[ k ];

			outputTextStream << "[" << k << "] (" << x << "," << y << ") ((" << x << "," << yy << ")): " << r << "\n";

			++k;
		}
	}

	outputFile.close();
	return true;
}

bool Image1f::savePFM( QString filename )
{
	int w = width();
	int h = height();

	// use "wb" binary mode to ensure that on Windows,
	// newlines in the header are written out as '\n'
	FILE* fp = fopen( qPrintable( filename ), "wb" );
	if( fp == NULL )
	{
		return false;
	}

	// write header
	fprintf( fp, "Pf\n%d %d\n-1\n", w, h );

	// write data
	for( int y = 0; y < h; ++y )
	{
		int yy = h - y - 1;
		float* row = &( m_data[ yy * w ] );
		fwrite( row, w * sizeof( float ), 1, fp );
	}

	fclose( fp );
	return true;
}
