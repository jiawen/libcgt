#pragma once

#include <vector>

#include <vecmath/Vector2f.h>
#include <vecmath/Matrix3f.h>

#include <FloatMatrix.h>

class PointLineICP
{
public:

	PointLineICP( int maxNumIterations = 6, float epsilon = 0.1f );

	// returns the rigid transformation M that best transforms the
	// *source points* to align with the destination points.
	//
	// This version assumes perfect correspondence
	//
	// TODO: pass an initialGuess as Matrix4f& and mutate in place?
	// or return a Maybe?
	bool align( const std::vector< Vector2f >& srcPoints,
		const std::vector< Vector2f >& dstPoints, const std::vector< Vector2f >& dstNormals,
		const Matrix3f& initialGuess,
		Matrix3f& outputSrcToDestination );

private:

	float updateSourcePointsAndEvaluateEnergy( const Matrix3f& incremental,
		const std::vector< Vector2f >& dstPoints, const std::vector< Vector2f >& dstNormals,
		std::vector< Vector2f >& srcPoints2 );

	int m_maxNumIterations;
	float m_epsilon;

	FloatMatrix m_A;
	FloatMatrix m_b;

};