#pragma once

#include <vecmath/Matrix3f.h>
#include <vecmath/Matrix4f.h>
#include <vecmath/Vector3f.h>

namespace libcgt { namespace core { namespace vecmath {

// TODO(jiawen): templatize this for floats and doubles.
// TODO(jiawen): store a quaternion
class EuclideanTransform
{
public:

    Matrix3f rotation;
    Vector3f translation;

    // User-defined conversion to Matrix4f.
    operator Matrix4f() const;
};

// Same as compose().
EuclideanTransform operator * ( const EuclideanTransform& second,
    const EuclideanTransform& first );

// Compose two Euclidean transformations. The resulting Euclidean transformation
// is equivalent to first applying "first", then applying "second".
//   R2 * (R1 * x + t1) + t2
// = R2 * R1 * x + R2 * t1 + t2
EuclideanTransform compose( const EuclideanTransform& second,
    const EuclideanTransform& first );

// Given a Euclidean transformation q = R p + t,
// Returns its inverse:
// p = R^T * (q - t)
//   = R^T q - R^T t
//   = R^T q + R^T * (-t)
// rInv = R^T
// tInv = R^T * (-t)
EuclideanTransform inverse( const EuclideanTransform& et );

// Construct the identity transformation.
EuclideanTransform makeIdentity();

// Apply the transformation et to the point p: q = R p + t.
Vector3f transformPoint( const EuclideanTransform& et, const Vector3f& p );

// Apply the transformation et to the vector v: w = R v.
Vector3f transformVector( const EuclideanTransform& et, const Vector3f& v );

// Convert a Euclidean transformation (R,t) from OpenCV conventions to OpenGL
// conventions.
//
// OpenCV conventions are right handed: x: right, y: down, z: into the screen.
// OpenGL conventions are right handed: x: right, y: up, z: out of the screen.
// In both conventions, R is applied first, then t: q = R * p + t.
//
// Given an OpenCV transformation A_cv_4x4 = (R_cv, t_cv), the OpenGL
// transformation is A_gl_4x4 = rx * A_cv_4x4 * rx, where rx is a rotation by
// 180 degrees around the x axis.
//
EuclideanTransform glFromCV( const EuclideanTransform& cv );

} } } // vecmath, core, libcgt
