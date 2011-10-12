#ifndef IMAGE_4F_H
#define IMAGE_4F_H

#include <QImage>
#include <QString>
#include <QVector>

#include <common/Reference.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector4i.h>
#include <vecmath/Vector4f.h>
#include <math/MersenneTwister.h>

class Image4f
{
public:

	// loads a 3-channel little-endian PFM from "filename" (magic = "PF")
	// returns NULL if it loading was unsuccessful
	static Reference< Image4f > loadPFM( QString filename );

	// loads a 4-channel litltle-endian PFM from "filename" (magic = "PF4")
	// returns NULL if it loading was unsuccessful
	static Reference< Image4f > loadPFM4( QString filename );

	Image4f(); // default constructor creates the null image
	Image4f( QString filename );
	
	Image4f( int width, int height, const Vector4f& fill = Vector4f( 0, 0, 0, 0 ) );
	Image4f( const Vector2i& size, const Vector4f& fill = Vector4f( 0, 0, 0, 0 ) );
	
	Image4f( const Image4f& copy );
	Image4f( Reference< Image4f > copy );

	bool isNull() const;

	int width() const;
	int height() const;
	Vector2i size() const;

	float* pixels();

	Vector4f pixel( int x, int y ) const;
	Vector4f pixel( const Vector2i& xy ) const;
	void setPixel( int x, int y, const Vector4f& pixel );
	void setPixel( const Vector2i& xy, const Vector4f& pixel );

	// pixel in [0,255]
	void setPixel( int x, int y, const Vector4i& pixel );
	void setPixel( const Vector2i& xy, const Vector4i& pixel );
	
	Reference< Image4f > flipUD();

	Vector4f bilinearSample( float x, float y ) const;
	Vector4f bilinearSample( const Vector2f& xy ) const;

	QImage toQImage();
	bool savePNG( QString filename );

	bool saveTXT( QString filename );

	// writes out a standard 3-component floating point image with no alpha channel
	// in little endian format
	bool savePFM( QString filename );

	// writes out a *non-standard* 4-component floating point image that contains the alpha channel
	// in little endian format
	bool savePFM4( QString filename );

private:

	int m_width;
	int m_height;
	QVector< float > m_data;

};

#endif // IMAGE_4F_H
