#pragma once

#include <QtGlobal>
#include <QImage>

#include "common/Array2D.h"
#include "common/BasicTypes.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"

class Image1i
{
public:

	Image1i(); // default constructor creates the null image

	Image1i( int width, int height, int32 fill = 0 );
	Image1i( const Vector2i& size, int32 fill = 0 );
	Image1i( const Image1i& copy );

	bool isNull() const;

	int width() const;
	int height() const;
	Vector2i size() const;

	const int32* pixels() const;
	int32* pixels();

	int32* rowPointer( int y );

	int32 pixel( int x, int y ) const;
	int32 pixel( const Vector2i& xy ) const;

	void setPixel( int x, int y, qint32 pixel );
	void setPixel( const Vector2i& xy, qint32 pixel );

	Image1i flipUD();

	int32 bilinearSample( float x, float y ) const;

	// Returns a 4-component QImage
	// with RGB clamped to [0,255] and alpha = 255
	QImage toQImage();

	void savePNG( QString filename );
	bool saveTXT( QString filename );

private:

	int m_width;
	int m_height;
	Array2D< int32 > m_data;

};
