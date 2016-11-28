#pragma once

#include <common/ArrayView.h>
#include <math/Random.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector4f.h>

namespace libcgt { namespace core { namespace imageproc {

template< typename T >
void createCheckerboard( Array2DWriteView< T > image,
    const Vector2i& checkerSize,
    const T& blackColor = T( 0 ), const T& whiteColor = T( 1 ) );

void createRandom( Array2DWriteView< float > image, Random& random );
void createRandom( Array2DWriteView< Vector4f > image, Random& random );

} } } // imageproc, core, libcgt

template< typename T >
void libcgt::core::imageproc::createCheckerboard( Array2DWriteView< T > image,
    const Vector2i& checkerSize, const T& blackColor, const T& whiteColor )
{
    int nBlocksX = 1 + image.width() / checkerSize.x;
    int nBlocksY = 1 + image.height() / checkerSize.y;

    bool rowIsWhite = true;
    bool isWhite;

    for( int by = 0; by < nBlocksY; ++by )
    {
        isWhite = rowIsWhite;
        for( int bx = 0; bx < nBlocksX; ++bx )
        {
            for( int y = by * checkerSize.y; ( y < ( by + 1 ) * checkerSize.y ) && ( y < image.height() ); ++y )
            {
                for( int x = bx * checkerSize.x; ( x < ( bx + 1 ) * checkerSize.x ) && ( x < image.width() ); ++x )
                {
                    image[ { x, y } ] = isWhite ? whiteColor : blackColor;
                }
            }

            isWhite = !isWhite;
        }
        rowIsWhite = !rowIsWhite;
    }
}
