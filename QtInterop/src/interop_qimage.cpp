#include "interop_qimage.h"

#include <QImage>

Array2DView< uint8x3 > libcgt::qtinterop::qimage::viewRGB32AsBGR( QImage& q )
{
    if( q.format() != QImage::Format_RGB32 )
    {
        return Array2DView< uint8x3 >();
    }

    return Array2DView< uint8x3 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

Array2DView< uint8x4 > libcgt::qtinterop::qimage::viewRGB32AsBGRX( QImage& q )
{
    if( q.format() != QImage::Format_RGB32 )
    {
        return Array2DView< uint8x4 >();
    }

    return Array2DView< uint8x4 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

Array2DView< uint8x4 > libcgt::qtinterop::qimage::viewARGB32AsBGRA( QImage& q )
{
    if( q.format() != QImage::Format_ARGB32 )
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
