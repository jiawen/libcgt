#include "libcgt/opencv_interop/ArrayUtils.h"

namespace libcgt { namespace opencv_interop {

const cv::Mat array2DViewAsCvMat( Array2DReadView< uint8x3 > view )
{
    // Use const_cast because OpenCV wants a void* even if it's const.
    return cv::Mat
    (
        view.height(), view.width(), CV_8UC3,
        const_cast< uint8x3* >( view.pointer() )
    );
}

cv::Mat array2DViewAsCvMat( Array2DWriteView< uint8x3 > view )
{
    return cv::Mat
    (
        view.height(), view.width(), CV_8UC3,
        view.pointer()
    );
}

} } // opencv_interop, libcgt
