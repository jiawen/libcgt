namespace libcgt { namespace core { namespace imageproc {

inline float toFloat( uint8_t x )
{
    return x / 255.f;
}

inline Vector2f toFloat( uint8x2 v )
{
    return{ toFloat( v.x ), toFloat( v.y ) };
}

inline Vector3f toFloat( uint8x3 v )
{
    return{ toFloat( v.x ), toFloat( v.y ), toFloat( v.z ) };
}

inline Vector4f toFloat( uint8x4 v )
{
    return{ toFloat( v.x ), toFloat( v.y ), toFloat( v.z ), toFloat( v.w ) };
}

inline uint8_t toUInt8( float x )
{
    return static_cast< uint8_t >( 255.0f * x + 0.5f );
}

inline uint8x2 toUInt8( Vector2f v )
{
    return{ toUInt8( v.x ), toUInt8( v.y ) };
}

inline uint8x3 toUInt8( Vector3f v )
{
    return{ toUInt8( v.x ), toUInt8( v.y ), toUInt8( v.z ) };
}

inline uint8x4 toUInt8( Vector4f v )
{
    return{ toUInt8( v.x ), toUInt8( v.y ), toUInt8( v.z ), toUInt8( v.w ) };
}

inline float toFloat( int8_t x )
{
    return std::max( x / 127.0f, -1.0f );
}

inline Vector2f toFloat( int8x2 v )
{
    return{ toFloat( v.x ), toFloat( v.y ) };
}

inline Vector3f toFloat( int8x3 v )
{
    return{ toFloat( v.x ), toFloat( v.y ), toFloat( v.z ) };
}

inline Vector4f toFloat( int8x4 v )
{
    return{ toFloat( v.x ), toFloat( v.y ), toFloat( v.z ), toFloat( v.w ) };
}

inline int8_t toSInt8( float x )
{
    return static_cast< int8_t >( x * 127.0f );
}

inline int8x2 toSInt8( Vector2f v )
{
    return{ toSInt8( v.x ), toSInt8( v.y ) };
}

inline int8x3 toSInt8( Vector3f v )
{
    return{ toSInt8( v.x ), toSInt8( v.y ), toSInt8( v.z ) };
}

inline int8x4 toSInt8( Vector4f v )
{
    return{ toSInt8( v.x ), toSInt8( v.y ), toSInt8( v.z ), toSInt8( v.w ) };
}

inline float saturate( float x )
{
    if( x < 0 )
    {
        x = 0;
    }
    else if( x > 1 )
    {
        x = 1;
    }
    return x;
}

inline Vector2f saturate( Vector2f v )
{
    return
    {
        saturate( v.x ),
        saturate( v.y )
    };
}

inline Vector3f saturate( Vector3f v )
{
    return
    {
        saturate( v.x ),
        saturate( v.y ),
        saturate( v.z )
    };
}

inline Vector4f saturate( Vector4f v )
{
    return Vector4f
    {
        saturate( v.x ),
        saturate( v.y ),
        saturate( v.z ),
        saturate( v.w )
    };
}

inline float rgbToLuminance( Vector3f rgb )
{
    return( 0.3279f * rgb.x + 0.6557f * rgb.y + 0.0164f * rgb.z );
}

inline uint8_t rgbToLuminance( uint8x3 rgb )
{
    uint16_t lr = (  84 * static_cast< uint16_t >( rgb.x ) ) >> 8;
    uint16_t lg = ( 167 * static_cast< uint16_t >( rgb.y ) ) >> 8;
    uint16_t lb = (   4 * static_cast< uint16_t >( rgb.z ) ) >> 8;
    return static_cast< uint8_t >( lr + lg + lb );
}

inline Vector3f rgb2xyz( Vector3f rgb )
{
    float rOut = ( rgb.x > 0.04045f ) ?
        pow( ( rgb.x + 0.055f ) / 1.055f, 2.4f ) :
        rgb.x / 12.92f;
    float gOut = ( rgb.y > 0.04045 ) ?
        pow( ( rgb.y + 0.055f ) / 1.055f, 2.4f ) :
        rgb.y / 12.92f;
    float bOut = ( rgb.z > 0.04045f ) ?
        pow( ( rgb.z + 0.055f ) / 1.055f, 2.4f ) :
        rgb.z / 12.92f;

    Vector3f rgbOut = 100 * Vector3f( rOut, gOut, bOut );

    return
    {
        Vector3f::dot( rgbOut, Vector3f( 0.4124f, 0.3576f, 0.1805f ) ),
        Vector3f::dot( rgbOut, Vector3f( 0.2126f, 0.7152f, 0.0722f ) ),
        Vector3f::dot( rgbOut, Vector3f( 0.0193f, 0.1192f, 0.9505f ) )
    };
}

inline Vector3f xyz2lab( Vector3f xyz, const Vector3f& xyzRef,
    float epsilon, float kappa )
{
    Vector3f xyzNormalized = xyz / xyzRef;

    float fx = ( xyzNormalized.x > epsilon ) ?
        pow( xyzNormalized.x, 1.f / 3.f ) :
        ( ( kappa * xyzNormalized.x + 16.f ) / 116.f );
    float fy = ( xyzNormalized.y > epsilon ) ?
        pow( xyzNormalized.y, 1.f / 3.f ) :
        ( ( kappa * xyzNormalized.y + 16.f ) / 116.f );
    float fz = ( xyzNormalized.z > epsilon ) ?
        pow( xyzNormalized.z, 1.f / 3.f ) :
        ( ( kappa * xyzNormalized.z + 16.f ) / 116.f );

    return Vector3f
    (
        ( 116.f * fy ) - 16.f,
        500.f * ( fx - fy ),
        200.f * ( fy - fz )
    );
}

inline Vector3f rgb2lab( Vector3f rgb )
{
    return xyz2lab( rgb2xyz( rgb ) );
}

inline Vector3f hsv2rgb( Vector3f hsv )
{
    float h = hsv.x;
    float s = hsv.y;
    float v = hsv.z;

    float r;
    float g;
    float b;

    h *= 360.f;
    int i;
    float f, p, q, t;

    if( s == 0 )
    {
        // achromatic (grey)
        return Vector3f( v, v, v );
    }
    else
    {
        h /= 60.f; // sector 0 to 5
        i = libcgt::core::math::floorToInt( h );
        f = h - i; // factorial part of h
        p = v * ( 1.f - s );
        q = v * ( 1.f - s * f );
        t = v * ( 1.f - s * ( 1.f - f ) );

        switch( i )
        {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }

        return Vector3f( r, g, b );
    }
}

inline Vector4f hsv2rgb( Vector4f hsva )
{
    return Vector4f( hsv2rgb( hsva.xyz ), hsva.w );
}

inline Vector4f jet( float x )
{
    float fourX = 4 * x;
    float r = std::min( fourX - 1.5f, -fourX + 4.5f );
    float g = std::min( fourX - 0.5f, -fourX + 3.5f );
    float b = std::min( fourX + 0.5f, -fourX + 2.5f );

    return saturate( Vector4f( r, g, b, 1 ) );
}

inline float logL( float l )
{
    const float logMin = log( LOG_LAB_EPSILON );
    const float logRange = log( 100 + LOG_LAB_EPSILON ) - logMin;

    float logL = log( l + LOG_LAB_EPSILON );

    // scale between 0 and 1
    float logL_ZO = ( logL - logMin ) / logRange;

    // scale between 0 and 100
    return 100.f * logL_ZO;
}

inline float expL( float ll )
{
    const float logMin = log( LOG_LAB_EPSILON );
    const float logRange = log( 100 + LOG_LAB_EPSILON ) - logMin;

    // scale between 0 and 1
    float logL_ZO = ll / 100.f;
    // bring back to log scale
    float logL = logL_ZO * logRange + logMin;

    // exponentiate
    return exp( logL ) - LOG_LAB_EPSILON;
}

} } } // imageproc, core, libcgt
