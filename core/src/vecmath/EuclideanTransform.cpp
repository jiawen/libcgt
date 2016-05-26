#include "EuclideanTransform.h"

namespace libcgt { namespace core { namespace vecmath {

EuclideanTransform::operator Matrix4f() const
{
    Matrix4f m;
    m.setSubmatrix3x3( 0, 0, rotation );
    m.setCol(3, { translation, 1 } );
    return m;
}

EuclideanTransform operator * ( const EuclideanTransform& second,
    const EuclideanTransform& first )
{
    return compose( second, first );
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
    return
    {
        et.rotation.transposed(),
        et.rotation.transposed() * ( -et.translation )
    };
}

EuclideanTransform makeIdentity()
{
    return
    {
        Matrix3f::identity(),
        Vector3f()
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
    Matrix4f glMatrix = rx * cv * rx;
    return
    {
        glMatrix.getSubmatrix3x3(),
        glMatrix.getCol(3).xyz
    };
}

} } } // vecmath, core, libcgt
