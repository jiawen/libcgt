#include "imageproc/Compositing.h"

#include "color/ColorUtils.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4f.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
Image4f Compositing::compositeOver( const Image4f& foreground, const Image4f& background )
{
	Image4f composite( foreground.size() );

	for( int y = 0; y < foreground.height(); ++y )
	{
		for( int x = 0; x < foreground.width(); ++x )
		{
			Vector4f f = foreground.pixel( x, y );
			Vector4f b = background.pixel( x, y );

			float fa = f.w;
			float ba = b.w;

			Vector3f compositeColor = fa * f.xyz() + ( 1.f - fa ) * ( b.xyz() );
			float compositeAlpha = fa + ba * ( 1 - fa );

			composite.setPixel( x, y, Vector4f( compositeColor, compositeAlpha ) );
		}
	}

	return composite;
}

// static
Image4f Compositing::extractBackgroundColor( const Image4f& composite, const Image4f& foreground )
{
	Image4f background( composite.size() );

	// red channel:
	// c_r = f_a * f_r + ( 1 - f_a ) * b_r
	// b_r = ( c_r - f_a * f_r ) / ( 1 - f_a )
	//
	// alpha channel:
	// c_a = f_a + b_a * ( 1 - f_a )
	// b_a = ( c_a - f_a ) / ( 1 - f_a )

	for( int y = 0; y < composite.height(); ++y )
	{
		for( int x = 0; x < composite.width(); ++x )
		{
			Vector4f cRGBA = composite.pixel( x, y );
			Vector4f fRGBA = foreground.pixel( x, y );
			
			Vector4f bRGBA = extractBackgroundColor( cRGBA, fRGBA );
			background.setPixel( x, y, bRGBA );
		}
	}

	return background;
}

// static
Image4ub Compositing::extractBackgroundColor( const Image4ub& composite, const Image4ub& foreground )
{
	Image4ub background( composite.size() );

	for( int y = 0; y < composite.height(); ++y )
	{
		for( int x = 0; x < composite.width(); ++x )
		{
			Vector4i cRGBA = composite.pixel( x, y );
			Vector4f cRGBAFloat = Vector4f( ColorUtils::intToFloat( cRGBA ) );
			Vector4f fRGBA = ColorUtils::intToFloat( foreground.pixel( x, y ) );

			Vector4f bRGBAFloat = extractBackgroundColor( cRGBAFloat, fRGBA );
			Vector4i bRGBA = ColorUtils::floatToInt( bRGBAFloat );
			background.setPixel( x, y, bRGBA );
		}
	}

	return background;
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
Vector4f Compositing::extractBackgroundColor( const Vector4f& composite, const Vector4f& foreground )
{
	// red channel:
	// c_r = f_a * f_r + ( 1 - f_a ) * b_r
	// b_r = ( c_r - f_a * f_r ) / ( 1 - f_a )
	//
	// alpha channel:
	// c_a = f_a + b_a * ( 1 - f_a )
	// b_a = ( c_a - f_a ) / ( 1 - f_a )

	Vector3f cRGB = composite.xyz();
	Vector3f fRGB = foreground.xyz();

	Vector4f bRGBA;

	float fa = foreground.w;
	if( fa < 1.f )
	{
		float ca = composite.w;

		Vector3f bRGB = ( cRGB - fa * fRGB ) / ( 1.f - fa );
		float ba = ( ca - fa ) / ( 1.f - fa );
		
#if 0
		if( bRGB.x() < 0 || bRGB.y() < 0 || bRGB.z() < 0 ||
			bRGB.x() > 1 || bRGB.y() > 1 || bRGB.z() > 1 )
		{
			bRGBA = Vector4f( cRGB, 0.f );
		}
		else
		{
			bRGBA = Vector4f( bRGB, ba );
		}
#else
		bRGBA = Vector4f( bRGB, ba );
#endif
	}
	else // foreground alpha = 1, set the background to the composite color with alpha = 0
	{
		// bRGBA = Vector4f( 0.f, 0.f, 0.f, 0.f );
		bRGBA = Vector4f( cRGB, 0.f );
	}

	return bRGBA;
}
