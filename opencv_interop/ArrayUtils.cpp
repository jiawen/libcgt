#include "libcgt/opencv_interop/ArrayUtils.h"

namespace libcgt { namespace opencv_interop {

const cv::Mat array2DViewAsCvMat( Array2DReadView< uint8_t > view )
{
    assert( view.elementsArePacked() );
    if( !view.elementsArePacked() )
    {
        return cv::Mat();
    }

    // Use const_cast because OpenCV wants a void* even if it's const.
    return cv::Mat
    (
        view.height(), view.width(), CV_8UC1,
        const_cast< uint8_t* >( view.pointer() ),
        view.rowStrideBytes()
    );
}

cv::Mat array2DViewAsCvMat( Array2DWriteView< uint8_t > view )
{
    if( !view.elementsArePacked() )
    {
        return cv::Mat();
    }

    return cv::Mat
    (
        view.height(), view.width(), CV_8UC1,
        view.pointer(),
        view.rowStrideBytes()
    );
}

const cv::Mat array2DViewAsCvMat( Array2DReadView< uint8x3 > view )
{
    if( !view.elementsArePacked() )
    {
        return cv::Mat();
    }

    // Use const_cast because OpenCV wants a void* even if it's const.
    return cv::Mat
    (
        view.height(), view.width(), CV_8UC3,
        const_cast< uint8x3* >( view.pointer() ),
        view.rowStrideBytes()
    );
}

cv::Mat array2DViewAsCvMat( Array2DWriteView< uint8x3 > view )
{
    if( !view.elementsArePacked() )
    {
        return cv::Mat();
    }

    return cv::Mat
    (
        view.height(), view.width(), CV_8UC3,
        view.pointer(),
        view.rowStrideBytes()
    );
}

} } // opencv_interop, libcgt
