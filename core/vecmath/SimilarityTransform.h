#pragma once

#include "libcgt/core/vecmath/EuclideanTransform.h"

namespace libcgt { namespace core { namespace vecmath {

// TODO(jiawen): templatize this for floats and doubles.
// A Similarity transform, represented as uniform scale and a
// EuclideanTransform (a rigid body transformation).
//
// Note that the rotation is applied first, then scale, then translation.
// q = s.scale * e.rotation * p + e.translation.
//
// SimilarityTransform defaults to an identity transformation.
class SimilarityTransform
{
public:

    float scale = 1.0f;
    EuclideanTransform euclidean;

    SimilarityTransform() = default;
    // Scale only.
    explicit SimilarityTransform( float s );
    // Rotation only.
    explicit SimilarityTransform( const Matrix3f& r );
    // Translation only.
    explicit SimilarityTransform( const Vector3f& t );
    // Euclidean only.
    explicit SimilarityTransform( const EuclideanTransform& e );
    // Linear only.
    SimilarityTransform( float s, const Matrix3f& r );
    SimilarityTransform( float s, const Matrix3f& r, const Vector3f& t );
    SimilarityTransform( float s, const EuclideanTransform& e );

    // Assuming m is a composition of a scale, a rotation and a translation,
    // constructs the equivalent SimilarityTransform.
    static SimilarityTransform fromMatrix( const Matrix4f& m );

    // Matrix4f representation of this SimilarityTransform.
    Matrix4f asMatrix() const;

    // Returns the Matrix3f that transforms normals (as opposed to points or
    // vectors) as this SimilarityTransform does with transformNormal().
    Matrix3f normalMatrix() const;

    // The rotation part of this SimilarityTransform.
    const Matrix3f& rotation() const;
    Matrix3f& rotation();

    // The translation part of this SimilarityTransform.
    const Vector3f& translation() const;
    Vector3f& translation();
};

// Same as compose().
SimilarityTransform operator * ( const SimilarityTransform& second,
    const SimilarityTransform& first );

// Compose two Similarity transformations. The resulting Similarity
// transformation is equivalent to first applying "first", then applying
// "second".
//   S2 * R2 * (S1 * R1 * x + t1) + t2
// = (S2 * R2 * S1 * R1) * x + (S2 * R2) * t1 + t2
// = (S2 * S1) * (R2 * R1) * x + (S2 * R2 * t1 + t2)
SimilarityTransform compose( const SimilarityTransform& second,
    const SimilarityTransform& first );

// Given a Similarity transformation q = s R p + t,
// Returns its inverse:
// p = R^T * ((q - t) / s)
//   = (1/s) * R^T q - (1/s) * R^T t
//   = (1/s) * R^T q + (1/s) * R^T * (-t)
// sInv = 1/s
// rInv = R^T
// tInv = (1/s) * R^T * (-t)
SimilarityTransform inverse( const SimilarityTransform& st );

// Apply the transformation st to the point p: q = s R p + t.
Vector3f transformPoint( const SimilarityTransform& st, const Vector3f& p );

// Apply the transformation st to the normal n: m = (R / s) n.
Vector3f transformNormal( const SimilarityTransform& st, const Vector3f& n );

// Apply the transformation st to the vector v: w = s R v.
Vector3f transformVector( const SimilarityTransform& st, const Vector3f& v );

// Convert a Similarity transformation (s, R, t) from OpenCV conventions to
// OpenGL conventions.
//
// OpenCV conventions are right handed: x: right, y: down, z: into the screen.
// OpenGL conventions are right handed: x: right, y: up, z: out of the screen.
// In both conventions, R is applied first, then t: q = s R * p + t.
//
// Given an OpenCV transformation A_cv_4x4 = (R_cv, t_cv), the OpenGL
// transformation is A_gl_4x4 = rx * A_cv_4x4 * rx, where rx is a rotation by
// 180 degrees around the x axis.
SimilarityTransform glFromCV( const SimilarityTransform& cv );

// Convert a Similarity transformation (s, R, t) from OpenGL conventions to
// OpenCV conventions.
//
// OpenCV conventions are right handed: x: right, y: down, z: into the screen.
// OpenGL conventions are right handed: x: right, y: up, z: out of the screen.
// In both conventions, R is applied first, then t: q = s R * p + t.
//
// Given an OpenGL transformation A_gl_4x4 = (R_gl, t_gl), the OpenCV
// transformation is A_cv_4x4 = rx * A_gl_4x4 * rx, where rx is a rotation by
// 180 degrees around the x axis.
SimilarityTransform cvFromGL( const SimilarityTransform& gl );

} } } // vecmath, core, libcgt
