#include <imageproc/ColorMap.h>

#include <common/ArrayUtils.h>
#include <imageproc/ColorUtils.h>
#include <math/MathUtils.h>

using libcgt::core::arrayutils::map;
using libcgt::core::imageproc::colorutils::colorMapJet;
using libcgt::core::imageproc::colorutils::saturate;
using libcgt::core::imageproc::colorutils::toUInt8;
using libcgt::core::math::fraction;
using libcgt::core::math::rescale;

namespace libcgt { namespace core { namespace imageproc { namespace colormap {

void jet( Array2DView< const float > src, const Range1f& srcRange,
    Array2DView< uint8x4 > dst )
{
    map< float, uint8x4 >( src, dst,
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
    map< float, Vector4f >( src, dst,
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
    map< Vector4f, uint8x4 >( src, dst,
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
    map< Vector4f, Vector4f >( src, dst,
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

void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< uint8_t > dst )
{
    map< float, uint8_t >( src, dst,
        [&] ( float z )
        {
            float luma = saturate( rescale( z, srcRange, dstRange ) );
            return toUInt8( luma );
        }
    );
}

void linearRemapToLuminance( Array2DView< const float > src,
    const Range1f& srcRange, const Range1f& dstRange,
    Array2DView< uint8x4 > dst )
{
    map< float, uint8x4 >( src, dst,
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
    map< float, float >( src, dst,
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
    map< float, Vector4f >( src, dst,
        [&] ( float z )
        {
            float luma = saturate( rescale( z, srcRange, dstRange ) );
            Vector4f rgba( luma, luma, luma, 1.0f );
            return rgba;
        }
    );
}

} } } } // colormap, imageproc, core, libcgt
