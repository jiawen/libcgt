#include "interop_qimage.h"

#include <QImage>

Array2DView< uint8x4 > libcgt::qtinterop::qimage::viewOfQImageARGB32( QImage& q )
{
    if( q.format() != QImage::Format_ARGB32 &&
        q.format() != QImage::Format_RGB32 )
    {
        return Array2DView< uint8x4 >();
    }

    return Array2DView< uint8x4 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

QImage libcgt::qtinterop::qimage::wrapAsQImage( Array2DView< uint8x4 > view )
{
    if( view.isNull() || !view.elementsArePacked() )
    {
        return QImage();
    }

    return QImage( reinterpret_cast< uchar* >( view.pointer() ),
        view.width(), view.height(), view.rowStrideBytes(),
        QImage::Format_ARGB32 );
}

QImage libcgt::qtinterop::qimage::wrapAsQImage( Array2DView< uint8_t > view )
{
    if( view.isNull() || !view.elementsArePacked() )
    {
        return QImage();
    }

    QImage q( reinterpret_cast< uchar* >( view.pointer() ),
        view.width(), view.height(), view.rowStrideBytes(),
        QImage::Format_Indexed8 );
    q.setColorCount( 256 );
    for( int i = 0; i < 256; ++i )
    {
        q.setColor( i, qRgb( i, i, i ) );
    }
    return q;
}