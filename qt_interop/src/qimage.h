#pragma once

#include <common/Array2DView.h>
#include <common/BasicTypes.h>

class QImage;

namespace libcgt { namespace qt_interop {

    // View the raw bytes of a QImage as an Array2DView< uint8_t >.
    Array2DView< uint8_t > viewGrayscale8( QImage& q );

    // View the raw bytes of a QImage as an Array2DView< uint8x3 >.
    // QImage::Format_RGB32 corresponds to a uint8x3 as BGRX,
    // where X is ignored, because Qt considers pixels as integers:
    // 0xFFRRGGBB.
    //
    // Returns a null Array2DView if the format is not Format_RGB32.
    Array2DView< uint8x3 > viewRGB32AsBGR( QImage& q );

    // View the raw bytes of a QImage as an Array2DView< uint8x4 >.
    // QImage::Format_RGB32 corresponds to a uint8x3 as BGRX,
    // where X is ignored, because Qt considers pixels as integers:
    // 0xFFRRGGBB.
    //
    // Returns a null Array2DView if the format is not Format_RGB32.
    Array2DView< uint8x4 > viewRGB32AsBGRX( QImage& q );

    // View the raw bytes of a QImage as an Array2DView< uint8x4 >.
    // QImage::Format_ARGB32 corresponds to a uint8x4 as BGRA
    // because Qt considers pixels as integers: 0xAARRGGBB.
    //
    // Returns a null Array2DView if the format is not Format_ARGB32.
    Array2DView< uint8x4 > viewARGB32AsBGRA( QImage& q );

    // TODO: const view using const bits. Array2DView needs a wrapped const constructor.

    // Wrap a view as a QImage and does not take ownership.
    // If view's format is BGRA, QImage will interpret it as 0xAARRGGBB
    // and life will be good.
    //
    // However, if the view's format is RGBA, then QImage will interpret
    // it as 0xAABBGGRR: the alpha channel is in the right place, but
    // the red and blue color channels are flipped.
    //
    // view must have its elements packed.
    // view's rows do *not* have to be packed.
    QImage wrapAsQImage( Array2DView< uint8x4 > view );

    // Wrap a view as a QImage and does not take ownership.
    QImage wrapAsQImage( Array2DView< uint8_t > view );

} } // qt_interop, libcgt
