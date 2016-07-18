#include "geometry/BoxUtils.h"

#include <cassert>

namespace libcgt { namespace core { namespace geometry { namespace boxutils {

void writeTriangleListPositions( const Box3f& box,
    Array1DView< Vector4f > positions )
{
    writeTriangleListPositions( box, Matrix4f::identity(), positions );
}

void writeTriangleListPositions( const Box3f& box,
    const Matrix4f& worldFromBox, Array1DView< Vector4f > positions )
{
    // Hypercube corners.
    Array1D< Vector4f > pos = corners( box );
    Array1D< int > idx = solidTriangleListIndices();

    for( int i = 0; i < idx.size(); ++i )
    {
        positions[ i ] = worldFromBox * pos[ idx[ i ] ];
    }
}

void writeTriangleListNormals( const Box3f& box,
    Array1DView< Vector3f > normals )
{
    writeTriangleListNormals( box, Matrix4f::identity(), normals );
}

void writeTriangleListNormals( const Box3f& box,
    const Matrix4f& worldFromBox, Array1DView< Vector3f > normals )
{
    Array1D< Vector4f > positions( 36 );
    writeTriangleListPositions( box, worldFromBox, positions );

    for( int f = 0; f < 12; ++f )
    {
        Vector3f p0 = positions[ 3 * f ].xyz;
        Vector3f p1 = positions[ 3 * f + 1 ].xyz;
        Vector3f p2 = positions[ 3 * f + 2 ].xyz;
        Vector3f n = Vector3f::cross( p1 - p0, p2 - p0 ).normalized();
        normals[ 3 * f ] = n;
        normals[ 3 * f + 1 ] = n;
        normals[ 3 * f + 1 ] = n;
    }
}

void writeAxisAlignedSolidTextureCoordinates(
    Array1DView< Vector2f > vertexTextureCoordinates )
{
    Array1D< Vector2f > uv =
    {
        { 0, 0 },
        { 1, 0 },
        { 0, 1 },
        { 1, 1 }
    };

    Array1D< int > idx =
    {
        0, 1, 2, 2, 1, 3,
        0, 1, 2, 2, 1, 3,
        0, 1, 2, 2, 1, 3,
        0, 1, 2, 2, 1, 3,
        0, 1, 2, 2, 1, 3,
        0, 1, 2, 2, 1, 3
    };

    for( int i = 0; i < idx.size(); ++i )
    {
        vertexTextureCoordinates[ i ] = uv[ idx[ i ] ];
    }
}

void writeWireframe( const Box3f& box,
    Array1DView< Vector4f > vertexPositions )
{
    writeWireframe( box, Matrix4f::identity(), vertexPositions );
}

void writeWireframe( const Box3f& box, const Matrix4f& worldFromBox,
    Array1DView< Vector4f > vertexPositions )
{
    // Hypercube corners.
    Array1D< Vector4f > pos = corners( box );
    Array1D< int > idx = wireframeLineListIndices();

    for( int i = 0; i < idx.size(); ++i )
    {
        vertexPositions[ i ] = worldFromBox * pos[ idx[ i ] ];
    }
}

void writeWireframeGrid( const Box3f& box, const Vector3i& resolution,
    Array1DView< Vector4f > vertexPositions )
{
    writeWireframeGrid( box, resolution, Matrix4f::identity(),
        vertexPositions );
}

void writeWireframeGrid( const Box3f& box, const Vector3i& resolution,
    const Matrix4f& worldFromBox, Array1DView< Vector4f > vertexPositions )
{
    Vector3f delta = box.size / resolution;
    Vector3f dx( 1, 0, 0 );
    Vector3f dy( 0, 1, 0 );
    Vector3f dz( 0, 0, 1 );

    int k = 0;

    for( int z = 0; z < resolution.z + 1; ++z )
    {
        for( int y = 0; y < resolution.y + 1; ++y )
        {
            vertexPositions[ 2 * k ] = worldFromBox * Vector4f( box.origin + y * delta.y * dy + z * delta.z * dz, 1 );
            vertexPositions[ 2 * k + 1 ] = worldFromBox * Vector4f( box.origin + box.size.x * dx + y * delta.y * dy + z * delta.z * dz, 1 );

            ++k;
        }
    }

    for( int z = 0; z < resolution.z + 1; ++z )
    {
        for( int x = 0; x < resolution.x + 1; ++x )
        {
            vertexPositions[ 2 * k ] = worldFromBox * Vector4f( box.origin + x * delta.x * dx + z * delta.z * dz, 1 );
            vertexPositions[ 2 * k + 1 ] = worldFromBox * Vector4f( box.origin + x * delta.x * dx + box.size.y * dy + z * delta.z * dz, 1 );

            ++k;
        }
    }

    for( int y = 0; y < resolution.y + 1; ++y )
    {
        for( int x = 0; x < resolution.x + 1; ++x )
        {
            vertexPositions[ 2 * k ] = worldFromBox * Vector4f( box.origin + x * delta.x * dx + y * delta.y * dy, 1 );
            vertexPositions[ 2 * k + 1 ] = worldFromBox * Vector4f( box.origin + x * delta.x * dx + y * delta.y * dy + box.size.z * dz, 1 );

            ++k;
        }
    }
}

Box3f makeCube( const Vector3f& center, float sideLength )
{
    Vector3f side{ sideLength };
    return{ center - 0.5f * side, side };
}

Array1D< Vector4f > corners( const Box3f& b )
{
    Array1D< Vector4f > out( 8 );

    for( int i = 0; i < 8; ++i )
    {
        out[ i ] =
        {
            ( i & 1 ) ? b.maximum().x : b.minimum().x,
            ( i & 2 ) ? b.maximum().y : b.minimum().y,
            ( i & 4 ) ? b.maximum().z : b.minimum().z,
            1.0f
        };
    }

    return out;
}

Array1D< Vector3i > corners( const Box3i& b )
{
    Array1D< Vector3i > out( 8 );

    for( int i = 0; i < 8; ++i )
    {
        out[ i ] =
        {
            ( i & 1 ) ? b.maximum().x : b.minimum().x,
            ( i & 2 ) ? b.maximum().y : b.minimum().y,
            ( i & 4 ) ? b.maximum().z : b.minimum().z
        };
    }

    return out;
}

Array1D< Vector3f > cornerNormalsDiagonal( const Box3f& b )
{
    Vector3f center = b.center();
    Array1D< Vector4f > positions = corners( b );
    Array1D< Vector3f > out( 8 );

    for( int i = 0; i < 8; ++i )
    {
        out[ i ] = ( positions[ i ].xyz - center ).normalized();
    }

    return out;
}

Array1D< int > solidTriangleListIndices()
{
    return Array1D< int >
    {
        // front
        4, 5, 6,
        6, 5, 7,

        // right
        5, 1, 7,
        7, 1, 3,

        // back
        1, 0, 3,
        3, 0, 2,

        // left
        0, 4, 2,
        2, 4, 6,

        // bottom
        0, 1, 4,
        4, 1, 5,

        // top
        6, 7, 2,
        2, 7, 3
    };
}

Array1D< int > wireframeLineListIndices()
{
    return Array1D< int >
    {
        0, 1,
        2, 3,
        4, 5,
        6, 7,
        1, 3,
        5, 7,
        0, 2,
        4, 6,
        0, 4,
        2, 6,
        1, 5,
        3, 7
    };
}

} } } } // boxutils, geometry, core, libcgt
