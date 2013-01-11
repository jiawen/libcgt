#pragma once

#include <common/BasicTypes.h>

#include <vecmath/Vector3f.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Vector4i.h>

class ColorUtils
{
public:

	// the epsilon used when converting to the log domain
	// and then input is "luminance" from rgbToLuminance()
	// value is 1 / 256
	static const float LOG_LUMINANCE_EPSILON;

	// the epsilon used when converting to the log domain
	// and then input is the L channel from CIE-Lab
	// the value is ( 1 / 256 ) * ( 100 / 256 );
	static const float LOG_LAB_EPSILON;

	static int floatToInt( float f );
	static float intToFloat( int i );

	static Vector3i floatToInt( const Vector3f& f );
	static Vector3f intToFloat( const Vector3i& i );

	static ubyte4 floatToUnsignedByte( const Vector4f& f );
	static Vector4i floatToInt( const Vector4f& f );
	static Vector4f intToFloat( const Vector4i& i );

	static ubyte floatToUnsignedByte( float f );
	static float unsignedByteToFloat( ubyte ub );

	static float rgbToLuminance( float rgb[3] );
	static float rgbToLuminance( ubyte rgb[3] );

	static float rgba2luminance( float rgba[4] );
	static float rgba2luminance( ubyte rgba[4] );

	static Vector3f rgb2xyz( const Vector3f& rgb );	

	// xyzRef = float3( 95.047, 100, 108.883 )
	// epsilon = 216 / 24389 = 0.00856
	// kappa = 24389 / 27 = 903.2963
	static Vector3f xyz2lab( const Vector3f& xyz,
		const Vector3f& xyzRef = Vector3f( 95.047f, 100.f, 108.883f ),
		float epsilon = 216.f / 24389.f,
		float kappa = 24389.f / 27.f );

	static Vector3f rgb2lab( const Vector3f& rgb );

	static Vector3f hsv2rgb( const Vector3f& hsv );
	// alpha is preserved
	static Vector4f hsva2rgba( const Vector4f& hsva );

	// given x in [0,1], returns an RGBA color like MATLAB's "jet" colormap
	static Vector4f colorMapJet( float x );

	// returns the logarithm of the L channel of an Lab image
	// offset by LOG_LAB_EPSILON and rescaled between 0 and 100
	static float logL( float l );

	// returns the anti-logarithm of the L channel of an Lab image
	// offset by LOG_LAB_EPSILON and rescaled between 0 and 100
	static float expL( float ll );

	// clamps f to [0,1]
	static float saturate( float f );
	static Vector3f saturate( const Vector3f& v );
	static Vector4f saturate( const Vector4f& v );

	// clamps i to [0,255]
	static ubyte saturate( int i );
	static Vector3i saturate( const Vector3i& v );
	static Vector4i saturate( const Vector4i& v );

};
