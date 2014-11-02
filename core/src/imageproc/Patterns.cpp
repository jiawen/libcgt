#include "imageproc/Patterns.h"

#include <cstdint>

template< typename T >
void libcgt::core::imageproc::patterns::createCheckerboard( Array2DView< T > image,
	int checkerSizeX, int checkerSizeY,
    const T& blackColor, const T& whiteColor )
{
	int nBlocksX = 1 + image.width() / checkerSizeX;
	int nBlocksY = 1 + image.height() / checkerSizeY;

	bool rowIsWhite = true;
	bool isWhite;

	for( int by = 0; by < nBlocksY; ++by )
	{
		isWhite = rowIsWhite;
		for( int bx = 0; bx < nBlocksX; ++bx )
		{
            for( int y = by * checkerSizeY; ( y < ( by + 1 ) * checkerSizeY ) && ( y < image.height() ); ++y )
			{
				for( int x = bx * checkerSizeX; ( x < ( bx + 1 ) * checkerSizeX ) && ( x < image.width() ); ++x )
				{					
                    image( x, y ) = isWhite ? whiteColor : blackColor;					
				}
			}

			isWhite = !isWhite;
		}
		rowIsWhite = !rowIsWhite;
	}
}

void libcgt::core::imageproc::patterns::createRandom( Array2DView< float > image, Random& random )
{
    for( int i = 0; i < image.width() * image.height(); ++i )
    {
        image[ i ] = random.nextFloat();
    }
}

void libcgt::core::imageproc::patterns::createRandom( Array2DView< Vector4f > image, Random& random )
{
    for( int i = 0; i < image.width() * image.height(); ++i )
    {        
        image[ i ] = random.nextVector4f();
    }
}

// Explicit template instantiation.
template
void libcgt::core::imageproc::patterns::createCheckerboard< uint8_t >( Array2DView< uint8_t > image,
    int checkerSizeX, int checkerSizeY,
    const uint8_t& blackColor, const uint8_t& whiteColor );

template
void libcgt::core::imageproc::patterns::createCheckerboard< uint16_t >( Array2DView< uint16_t > image,
    int checkerSizeX, int checkerSizeY,
    const uint16_t& blackColor, const uint16_t& whiteColor );

template
void libcgt::core::imageproc::patterns::createCheckerboard< uint32_t >( Array2DView< uint32_t > image,
    int checkerSizeX, int checkerSizeY,
    const uint32_t& blackColor, const uint32_t& whiteColor );

template
void libcgt::core::imageproc::patterns::createCheckerboard< float >( Array2DView< float > image,
    int checkerSizeX, int checkerSizeY,
    const float& blackColor, const float& whiteColor );

template
void libcgt::core::imageproc::patterns::createCheckerboard< Vector2f >( Array2DView< Vector2f > image,
    int checkerSizeX, int checkerSizeY,
    const Vector2f& blackColor, const Vector2f& whiteColor );

template
void libcgt::core::imageproc::patterns::createCheckerboard< Vector3f >( Array2DView< Vector3f > image,
    int checkerSizeX, int checkerSizeY,
    const Vector3f& blackColor, const Vector3f& whiteColor );

template
void libcgt::core::imageproc::patterns::createCheckerboard< Vector4f >( Array2DView< Vector4f > image,
    int checkerSizeX, int checkerSizeY,
    const Vector4f& blackColor, const Vector4f& whiteColor );
