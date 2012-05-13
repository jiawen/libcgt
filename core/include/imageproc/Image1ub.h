#pragma once

#include <QImage>

#include "common/BasicTypes.h"
#include "common/Array2D.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"

class Image1ub
{
public:

	Image1ub(); // default constructor creates the null image

	Image1ub( int width, int height, ubyte fill = 0 );
	Image1ub( const Vector2i& size, ubyte fill = 0 );
	Image1ub( const Image1ub& copy );

	bool isNull() const;

	int width() const;
	int height() const;
	Vector2i size() const;

	ubyte* pixels();

	ubyte pixel( int x, int y ) const;
	ubyte pixel( const Vector2i& xy ) const;

	void setPixel( int x, int y, ubyte pixel );
	void setPixel( const Vector2i& xy, ubyte pixel );

	ubyte bilinearSample( float x, float y ) const;

	// returns a 4-channel QImage
	// with rgb = value and alpha = 255
	QImage toQImage(); 
	void savePNG( QString filename );

private:

	int m_width;
	int m_height;
	Array2D< ubyte > m_data;

};
