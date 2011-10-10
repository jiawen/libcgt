#ifndef IMAGE_1F_H
#define IMAGE_1F_H

#include <QImage>
#include <QString>

#include "common/Reference.h"
#include "common/ReferenceCountedArray.h"
#include "vecmath/Vector2i.h"

class Image1f
{
public:

	static Reference< Image1f > loadPFM( QString filename );

	Image1f(); // default constructor creates the null image
	
	Image1f( int width, int height, float fill = 0.f );
	Image1f( const Vector2i& size, float fill = 0.f );
	
	Image1f( const Image1f& copy );
	Image1f( Reference< Image1f > copy );

	bool isNull() const;

	int width() const;
	int height() const;
	Vector2i size() const;

	FloatArray pixels();

	float pixel( int x, int y ) const;
	float pixel( const Vector2i& xy ) const;
	void setPixel( int x, int y, float pixel );
	void setPixel( const Vector2i& xy, float pixel );

	// pixel in [0,255]
	void setPixel( int x, int y, int pixel );
	void setPixel( const Vector2i& xy, int pixel );

	Reference< Image1f > flipUD();

	float bilinearSample( float x, float y ) const;
	float bilinearSample( const Vector2f& xy ) const;

	QImage toQImage();
	bool savePNG( QString filename );
	bool saveTXT( QString filename );
	bool savePFM( QString filename );

private:

	int m_width;
	int m_height;
	FloatArray m_data;

};

#endif // IMAGE_1F_H
