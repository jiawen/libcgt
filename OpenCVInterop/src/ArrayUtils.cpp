#include "ArrayUtils.h"

namespace libcgt { namespace opencv_interop { namespace arrayutils {

const cv::Mat array2DViewAsCvMat( Array2DView< const uint8x3 > view )
{
    // Use const_cast because OpenCV wants a void* even if it's const.
    return cv::Mat
    (
        view.height(), view.width(), CV_8UC3,
        const_cast< uint8x3* >( view.pointer() )
    );
}

cv::Mat array2DViewAsCvMat( Array2DView< uint8x3 > view )
{
    return cv::Mat
    (
        view.height(), view.width(), CV_8UC3,
        view.pointer()
    );
}

} } } // arrayutils, opencv_interop, libcgt
