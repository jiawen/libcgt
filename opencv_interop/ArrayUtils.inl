namespace libcgt { namespace opencv_interop {

template< typename S >
Array2DWriteView< S > cvMatAsArray2DView( const cv::Mat& a )
{
    // stride is (step[1], step[0]) because cv::Mat is indexed
    // as (row, col) but stored row major.
    Vector2i size{ a.cols, a.rows };
    Vector2i stride
    {
        static_cast< int >( a.step[1] ),
        static_cast< int >( a.step[0] )
    };
    return Array2DWriteView< S >( a.data, size, stride );
}

template< typename S, typename T >
Array2DWriteView< S > cvMatAsArray2DView( const cv::Mat_< T >& a )
{
    // stride is (step[1], step[0]) because cv::Mat is indexed
    // as (row, col) but stored row major.
    Vector2i size{ a.cols, a.rows };
    Vector2i stride
    {
        static_cast< int >( a.step[1] ),
        static_cast< int >( a.step[0] )
    };
    return Array2DWriteView< S >( a.data, size, stride );
}

} } // opencv_interop, libcgt
