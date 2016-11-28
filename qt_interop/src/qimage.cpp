#include "qimage.h"

#include <QImage>

namespace libcgt { namespace qt_interop {

#if( QT_VERSION >= QT_VERSION_CHECK( 5, 5, 0 ) )
Array2DWriteView< uint8_t > viewGrayscale8( QImage& q )
{
    if( q.format() != QImage::Format_Grayscale8 )
    {
        return Array2DWriteView< uint8_t >();
    }

    return Array2DWriteView< uint8_t >( q.bits(),
        { q.width(), q.height() }, { 1, q.bytesPerLine() } );
}
#endif

Array2DWriteView< uint8x3 > viewRGB32AsBGR( QImage& q )
{
    if( q.format() != QImage::Format_RGB32 )
    {
        return Array2DWriteView< uint8x3 >();
    }

    return Array2DWriteView< uint8x3 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

Array2DWriteView< uint8x4 > viewRGB32AsBGRX( QImage& q )
{
    if( q.format() != QImage::Format_RGB32 )
    {
        return Array2DWriteView< uint8x4 >();
    }

    return Array2DWriteView< uint8x4 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

Array2DWriteView< uint8x4 > viewARGB32AsBGRA( QImage& q )
{
    if( q.format() != QImage::Format_ARGB32 )
    {
        return Array2DWriteView< uint8x4 >();
    }

    return Array2DWriteView< uint8x4 >( q.bits(),
        { q.width(), q.height() }, { 4, q.bytesPerLine() } );
}

QImage wrapAsQImage( Array2DWriteView< uint8x4 > view )
{
    if( view.isNull() || !view.elementsArePacked() )
    {
        return QImage();
    }

    return QImage( reinterpret_cast< uchar* >( view.pointer() ),
        view.width(), view.height(), view.rowStrideBytes(),
        QImage::Format_ARGB32 );
}

#if( QT_VERSION >= QT_VERSION_CHECK( 5, 5, 0 ) )
QImage wrapAsQImage( Array2DWriteView< uint8_t > view )
{
    if( view.isNull() || !view.elementsArePacked() )
    {
        return QImage();
    }

    return QImage( reinterpret_cast< uchar* >( view.pointer() ),
        view.width(), view.height(), view.rowStrideBytes(),
        QImage::Format_Grayscale8 );
}
#endif

} } // qt_interop, libcgt
