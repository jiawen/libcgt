#include "qimage.h"

#include <QImage>

namespace libcgt
{
namespace qt_interop
{
namespace qimage
{
Array2DView< uint8x3 > viewRGB32AsBGR( QImage& q )
{
    if( q.format() != QImage::Format_RGB32 )
    {
        return Array2DView< uint8x3 >();
    }

    return Array2DView< uint8x3 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

Array2DView< uint8x4 > viewRGB32AsBGRX( QImage& q )
{
    if( q.format() != QImage::Format_RGB32 )
    {
        return Array2DView< uint8x4 >();
    }

    return Array2DView< uint8x4 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

Array2DView< uint8x4 > viewARGB32AsBGRA( QImage& q )
{
    if( q.format() != QImage::Format_ARGB32 )
    {
        return Array2DView< uint8x4 >();
    }

    return Array2DView< uint8x4 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

QImage wrapAsQImage( Array2DView< uint8x4 > view )
{
    if( view.isNull() || !view.elementsArePacked() )
    {
        return QImage();
    }

    return QImage( reinterpret_cast< uchar* >( view.pointer() ),
        view.width(), view.height(), view.rowStrideBytes(),
        QImage::Format_ARGB32 );
}
} // qimage
} // qt_interop
} // libcgt
