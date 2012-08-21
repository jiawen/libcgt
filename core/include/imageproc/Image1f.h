#pragma once

#include "common/Array2D.h"
#include "vecmath/Vector2i.h"

class QImage;
class QString;

// TODO: move constructor and assignment operator
// TODO: images should be "views" on top of Array2Ds
// with bilinear sampling and saving features

class Image1f
{
public:

	Image1f(); // default constructor creates the null image

	// Creates an Image1f from a PFM file	
	Image1f( QString filename );
	
	Image1f( int width, int height, float fill = 0.f );
	Image1f( const Vector2i& size, float fill = 0.f );
	
	Image1f( const Image1f& copy );

	bool isNull() const;

	int width() const;
	int height() const;
	Vector2i size() const;

	const float* pixels() const;
	float* pixels();

	float* rowPointer( int y );

	float pixel( int x, int y ) const;
	float pixel( const Vector2i& xy ) const;
	void setPixel( int x, int y, float pixel );
	void setPixel( const Vector2i& xy, float pixel );

	// pixel in [0,255]
	void setPixel( int x, int y, int pixel );
	void setPixel( const Vector2i& xy, int pixel );

	Image1f flipUD();

	float bilinearSample( float x, float y ) const;
	float bilinearSample( const Vector2f& xy ) const;

	// Clamps this Image1f to [0,1]
	// and returns a QImage with RGB in [0,255], and alpha = 1
	QImage toQImage();

	// ---- I/O ----
	bool load( QString filename );
	bool save( QString filename );
	
private:

	bool loadPFM( QString filename );
	bool savePNG( QString filename );
	bool saveTXT( QString filename );
	bool savePFM( QString filename );

	int m_width;
	int m_height;
	Array2D< float > m_data;

};
