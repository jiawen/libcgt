#include "libcgt/core/vecmath/EuclideanTransform.h"

#include "libcgt/core/math/MathUtils.h"
#include "libcgt/core/vecmath/Quat4f.h"

using libcgt::core::math::lerp;

namespace libcgt { namespace core { namespace vecmath {

EuclideanTransform::EuclideanTransform( const Matrix3f& r ) :
    rotation( r )
{

}

EuclideanTransform::EuclideanTransform( const Vector3f& t ) :
    translation( t )
{

}

EuclideanTransform::EuclideanTransform( const Matrix3f& r,
    const Vector3f& t ) :
    rotation( r ),
    translation( t )
{

}

// static
EuclideanTransform EuclideanTransform::fromMatrix( const Matrix4f& m )
{
    return
    {
        m.getSubmatrix3x3(),
        m.getCol( 3 ).xyz
    };
}

Matrix4f EuclideanTransform::asMatrix() const
{
    Matrix4f m;
    m.setSubmatrix3x3( 0, 0, rotation );
    m.setCol( 3, { translation, 1 } );
    return m;
}

EuclideanTransform operator * ( const EuclideanTransform& second,
    const EuclideanTransform& first )
{
    return compose( second, first );
}

bool operator == ( const EuclideanTransform& e0,
    const EuclideanTransform& e1 )
{
    return e0.rotation == e1.rotation && e0.translation == e1.translation;
}

bool operator != ( const EuclideanTransform& e0,
    const EuclideanTransform& e1 )
{
    return !( e0 == e1 );
}

EuclideanTransform compose( const EuclideanTransform& second,
    const EuclideanTransform& first )
{
    return
    {
        second.rotation * first.rotation,
        second.rotation * first.translation + second.translation
    };
}

EuclideanTransform inverse( const EuclideanTransform& et )
{
    Matrix3f ir = et.rotation.transposed();

    return
    {
        ir,
        ir * ( -et.translation )
    };
}

EuclideanTransform lerp( const EuclideanTransform& x,
    const EuclideanTransform& y, float t )
{
    Quat4f qx = Quat4f::fromRotationMatrix( x.rotation );
    Quat4f qy = Quat4f::fromRotationMatrix( y.rotation );

    return
    {
        Matrix3f::fromQuat( Quat4f::slerp( qx, qy, t ) ),
        ::lerp( x.translation, y.translation, t )
    };
}

Vector3f transformPoint( const EuclideanTransform& et, const Vector3f& p )
{
    return et.rotation * p + et.translation;
}

Vector3f transformVector( const EuclideanTransform& et, const Vector3f& v )
{
    return et.rotation * v;
}

EuclideanTransform glFromCV( const EuclideanTransform& cv )
{
    Matrix4f rx = Matrix4f::ROTATE_X_180;
    return EuclideanTransform::fromMatrix( rx * cv.asMatrix() * rx );
}

EuclideanTransform cvFromGL( const EuclideanTransform& gl )
{
    // The conversion back and forth is exactly the same.
    return glFromCV( gl );
}

} } } // vecmath, core, libcgt
