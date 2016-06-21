#include "SimilarityTransform.h"

#include <cassert>

namespace libcgt { namespace core { namespace vecmath {

SimilarityTransform::SimilarityTransform( float s ) :
    scale( s )
{

}

SimilarityTransform::SimilarityTransform( const Matrix3f& r ) :
    euclidean( r )
{

}

SimilarityTransform::SimilarityTransform( const Vector3f& t ) :
    euclidean( t )
{

}

SimilarityTransform::SimilarityTransform( const EuclideanTransform& e ) :
    euclidean( e )
{

}

SimilarityTransform::SimilarityTransform( float s, const Matrix3f& r ) :
    scale( s ),
    euclidean( r )
{

}

SimilarityTransform::SimilarityTransform( float s, const Matrix3f& r,
    const Vector3f& t ) :
    scale( s ),
    euclidean( r, t )
{

}

SimilarityTransform::SimilarityTransform( float s,
    const EuclideanTransform& e ) :
    scale( s ),
    euclidean( e )
{

}

// static
SimilarityTransform SimilarityTransform::fromMatrix( const Matrix4f& m )
{
    Matrix3f r = m.getSubmatrix3x3();
    float s = r.getRow( 0 ).norm();

    return
    {
        s,
        r / s,
        m.getCol( 3 ).xyz
    };
}

SimilarityTransform::operator Matrix4f() const
{
    Matrix4f m;
    m.setSubmatrix3x3( 0, 0, scale * euclidean.rotation );
    m.setCol( 3, { euclidean.translation, 1 } );
    return m;
}

const Matrix3f& SimilarityTransform::rotation() const
{
    return euclidean.rotation;
}

Matrix3f& SimilarityTransform::rotation()
{
    return euclidean.rotation;
}

const Vector3f& SimilarityTransform::translation() const
{
    return euclidean.translation;
}

Vector3f& SimilarityTransform::translation()
{
    return euclidean.translation;
}

SimilarityTransform operator * ( const SimilarityTransform& second,
    const SimilarityTransform& first )
{
    return compose( second, first );
}

SimilarityTransform compose( const SimilarityTransform& second,
    const SimilarityTransform& first )
{
    return
    {
        second.scale * first.scale,
        second.rotation() * first.rotation(),

        second.scale * second.rotation() * first.translation()
        + second.translation()
    };
}

SimilarityTransform inverse( const SimilarityTransform& st )
{
    assert( st.scale != 0.0f );

    float is = 1.0f / st.scale;
    Matrix3f ir = st.rotation().transposed();

    return
    {
        is,
        ir,
        is * ir * ( -st.translation() )
    };
}

Vector3f transformPoint( const SimilarityTransform& st, const Vector3f& p )
{
    return st.scale * st.rotation() * p + st.translation();
}

Vector3f transformVector( const SimilarityTransform& st, const Vector3f& v )
{
    return st.scale * st.rotation() * v;
}

SimilarityTransform glFromCV( const SimilarityTransform& cv )
{
    Matrix4f rx = Matrix4f::ROTATE_X_180;
    return SimilarityTransform::fromMatrix( rx * cv * rx );
}

SimilarityTransform cvFromGL( const SimilarityTransform& gl )
{
    // The conversion back and forth is exactly the same.
    return glFromCV( gl );
}

} } } // vecmath, core, libcgt
