#include "libcgt/cuda/ArrayUtils.h"

#include "libcgt/core/common/Array1D.h"
#include "libcgt/core/common/Array2D.h"
#include "libcgt/core/common/Array3D.h"
#include "libcgt/core/common/ArrayUtils.h"

// TODO: Use templates, this is silly.
// Make a template to convert cuda types into vecmath types.

using libcgt::core::arrayutils::cast;

namespace libcgt { namespace cuda {

bool saveTXT( Array1DReadView< int3 > view, const std::string& filename )
{
    return saveTXT( cast< Vector3i >( view ), filename );
}

bool saveTXT( Array2DReadView< float2 > view, const std::string& filename )
{
    return saveTXT( cast< Vector2f >( view ), filename );
}

bool saveTXT( Array2DReadView< float4 > view, const std::string& filename )
{
    return saveTXT( cast< Vector4f >( view ), filename );
}

bool saveTXT( Array2DReadView< uchar4 > view, const std::string& filename )
{
    return saveTXT( cast< uint8x4 >( view ), filename );
}

bool saveTXT( Array3DReadView< ushort2 > view, const std::string& filename )
{
    return saveTXT( cast< uint16x2 >( view ), filename );
}

bool saveTXT( Array3DReadView< int2 > view, const std::string& filename )
{
    return saveTXT( cast< Vector2i >( view ), filename );
}

bool saveTXT( Array3DReadView< int3 > view, const std::string& filename )
{
    return saveTXT( cast< Vector3i >( view ), filename );
}

bool saveTXT( Array3DReadView< int4 > view, const std::string& filename )
{
    return saveTXT( cast< Vector4i >( view ), filename );
}

bool saveTXT( const DeviceArray1D< int3 >& view, const std::string& filename )
{
    Array1D< int3 > h_array( view.length() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< float >& view, const std::string& filename )
{
    Array2D< float > h_array( view.size() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< float2 >& view, const std::string& filename )
{
    Array2D< float2 > h_array( view.size() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< float4 >& view, const std::string& filename )
{
    Array2D< float4 > h_array( view.size() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< uchar4 >& view, const std::string& filename )
{
    Array2D< uchar4 > h_array( view.size() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray3D< ushort2 >& view, const std::string& filename )
{
    Array3D< ushort2 > h_array( view.size() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray3D< int2 >& view, const std::string& filename )
{
    Array3D< int2 > h_array( view.size() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray3D< int3 >& view, const std::string& filename )
{
    Array3D< int3 > h_array( view.size() );
    copy( view, h_array.writeView() );
    return saveTXT( h_array, filename );
}

} } // cuda, libcgt
