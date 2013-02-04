#pragma once

#include <windows.h>
#include <Vfw.h>

#include <common/Array2D.h>
#include <math/Arithmetic.h>
#include <vecmath/Vector2i.h>

class AVIInputVideoStream
{
public:

	static AVIInputVideoStream* open( PAVIFILE pAVIFile, int streamIndex,
		int width, int height );

	virtual ~AVIInputVideoStream();

	int width() const;
	int height() const;
	Vector2i size() const;

	int numFrames() const;

	// can be different from container file
	float framesPerSecond() const;
	void framesPerSecondRational( int& numerator, int& denominator ) const;

	// Returns a view of frame frameIndex
	// in RGBA format with custom alpha
	// (Same as getFrameBGR and then swizzling it)
	//
	// Returns NULL if frameIndex is out of range
	Array2DView< ubyte4 > getFrameRGBA( int frameIndex, ubyte alpha = 255 );

	// Returns a view of frame frameIndex
	// in BGRA format with custom alpha
	// (Same as getFrameBGR and then swizzling it)
	//
	// Returns NULL if frameIndex is out of range
	Array2DView< ubyte4 > getFrameBGRA( int frameIndex, ubyte alpha = 255 );

	// Returns a view of frame frameIndex
	// in the natural BGR (24 bits per pixel, 8 bits per channel) format
	//
	// Returns NULL if frameIndex is out of range
	Array2DView< ubyte3 > getFrameBGR( int frameIndex );

private:

	AVIInputVideoStream();

	int m_width;
	int m_height;
	Array2D< ubyte4 > m_ubyte4Data;

	PAVISTREAM m_pStream;
	PGETFRAME m_pFrame;	
	AVISTREAMINFO m_info;

	int m_streamStartIndex;
	int m_nSamples;
};