#include <imageproc/ColorMap.h>

#include <common/ArrayUtils.h>
#include <imageproc/ColorUtils.h>
#include <math/MathUtils.h>

using libcgt::core::arrayutils::map;
using libcgt::core::imageproc::colorutils::colorMapJet;
using libcgt::core::imageproc::colorutils::saturate;
using libcgt::core::imageproc::colorutils::toUInt8;
using libcgt::core::math::clamp;
using libcgt::core::math::fraction;
using libcgt::core::math::rescale;

namespace libcgt { namespace core { namespace imageproc {

void jet( Array2DView< const float > src, const Range1f& srcRange,
    Array2DView< uint8x4 > dst )
{
    map( src, dst,
        [&] ( float z )
        {
            z = saturate( fraction( z, srcRange ) );
            return toUInt8( colorMapJet( z ) );
        }
    );
}

void jet( Array2DView< const float > src, const Range1f& srcRange,
    Array2DView< Vector4f > dst )
{
    map( src, dst,
        [&] ( float z )
        {
            z = saturate( fraction( z, srcRange ) );
            return colorMapJet( z );
        }
    );
}

void normalsToRGBA( Array2DView< const Vector4f > src,
    Array2DView< uint8x4 > dst )
{
    map( src, dst,
        [&] ( const Vector4f& normal )
        {
            Vector4f rgba;
            if( normal.w > 0 )
            {
                rgba.xyz = 0.5f * ( normal.xyz + Vector3f{ 1, 1, 1 } );
                rgba.w = 1;
            }
            return toUInt8( rgba );
        }
    );
}

void normalsToRGBA( Array2DView< const Vector4f > src,
    Array2DView< Vector4f > dst )
{
    map( src, dst,
        [&] ( const Vector4f& normal )
        {
            Vector4f rgba;
            if( normal.w > 0 )
            {
                rgba.xyz = 0.5f * ( normal.xyz + Vector3f{ 1, 1, 1 } );
                rgba.w = 1;
            }
            return rgba;
        }
    );
}

void linearRemapToLuminance( Array2DView< const uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    Array2DView< uint8_t > dst )
{
    float srcSize = static_cast< float >( srcRange.size );
    map( src, dst,
        [&] ( uint16_t z )
        {
            return static_cast< uint8_t >(
                clamp( rescale( z, srcRange, dstRange ), Range1i( 256 ) ) );
        }
    );
}

void linearRemapToLuminance( Array2DView< const uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    Array2DView< uint8x3 > dst )
{
    float srcSize = static_cast< float >( srcRange.size );
    map( src, dst,
        [&] ( uint16_t z )
        {
            uint8_t luma = static_cast< uint8_t >(
                clamp( rescale( z, srcRange, dstRange ), Range1i( 256 ) ) );
            return uint8x3{ luma, luma, luma };
        }
    );
}

void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< uint8_t > dst )
{
    map( src, dst,
        [&] ( float z )
        {
            float luma = saturate( rescale( z, srcRange, dstRange ) );
            return toUInt8( luma );
        }
    );
}

void linearRemapToLuminance( Array2DView< const uint16_t > src,
    const Range1i& srcRange, const Range1i& dstRange,
    uint8_t dstAlpha, Array2DView< uint8x4 > dst )
{
    float srcSize = static_cast< float >( srcRange.size );
    map( src, dst,
        [&] ( uint16_t z )
        {
            uint8_t luma = static_cast< uint8_t >(
                clamp( rescale( z, srcRange, dstRange ), Range1i( 256 ) ) );
            return uint8x4{ luma, luma, luma, dstAlpha };
        }
    );
}

void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< uint8x4 > dst )
{
    map( src, dst,
        [&] ( float z )
        {
            float luma = saturate( rescale( z, srcRange, dstRange ) );
            Vector4f rgba( luma, luma, luma, 1.0f );
            return toUInt8( rgba );
        }
    );
}

void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< float > dst )
{
    map( src, dst,
        [&] ( float z )
        {
            float luma = saturate( rescale( z, srcRange, dstRange ) );
            return luma;
        }
    );
}

void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< Vector4f > dst )
{
    map( src, dst,
        [&] ( float z )
        {
            float luma = saturate( rescale( z, srcRange, dstRange ) );
            Vector4f rgba( luma, luma, luma, 1.0f );
            return rgba;
        }
    );
}

} } } // imageproc, core, libcgt
