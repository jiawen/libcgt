#ifndef IMAGE_4UB_H
#define IMAGE_4UB_H

#include <QImage>
#include <QString>
#include <QtGlobal>
#include <QVector>

#include "common/Reference.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector4i.h"

class Image4f;

class Image4ub
{
public:

	Image4ub(); // default constructor creates the null image
	Image4ub( QString filename );

	Image4ub( int width, int height, const Vector4i& fill = Vector4i( 0, 0, 0, 0 ) );
	Image4ub( const Vector2i& size, const Vector4i& fill = Vector4i( 0, 0, 0, 0 ) );
	Image4ub( const Image4ub& copy );
	Image4ub( Reference< Image4ub > copy );

	bool isNull() const;

	int width() const;
	int height() const;
	Vector2i size() const;

	quint8* pixels();

	Vector4i pixel( int x, int y ) const;
	Vector4i pixel( const Vector2i& xy ) const;

	void setPixel( int x, int y, const Vector4i& pixel ); // values > 255 are saturated
	void setPixel( const Vector2i& xy, const Vector4i& pixel ); // values > 255 are saturated

	Vector4i bilinearSample( float x, float y ) const;

	QImage toQImage();
	void savePNG( QString filename );

private:

	int m_width;
	int m_height;
	QVector< quint8 > m_data;

};

#endif // IMAGE_4UB_H
