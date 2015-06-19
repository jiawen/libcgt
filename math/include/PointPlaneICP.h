#pragma once

#include <vector>

#include <vecmath/Vector3f.h>
#include <vecmath/Matrix4f.h>

#include <FloatMatrix.h>

class PointPlaneICP
{
public:

	PointPlaneICP( int maxNumIterations = 6, float epsilon = 0.1f );

	// returns the rigid transformation M that best transforms the
	// *source points* to align with the destination points.
	//
	// This version assumes perfect correspondence
	//
	// TODO: pass an initialGuess as Matrix4f& and mutate in place?
	// or return a Maybe?
	bool align( const std::vector< Vector3f >& srcPoints,
		const std::vector< Vector3f >& dstPoints, const std::vector< Vector3f >& dstNormals,
		const Matrix4f& initialGuess,
		Matrix4f& outputSrcToDestination );

private:

	float updateSourcePointsAndEvaluateEnergy( const Matrix4f& incremental,
		const std::vector< Vector3f >& dstPoints, const std::vector< Vector3f >& dstNormals,
		std::vector< Vector3f >& srcPoints2 );

	int m_maxNumIterations;
	float m_epsilon;

	FloatMatrix m_A;
	FloatMatrix m_b;

};