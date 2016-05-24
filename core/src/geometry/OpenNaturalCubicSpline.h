#pragma once

#include <vector>
#include <vecmath/Vector4f.h>

class OpenNaturalCubicSpline
{
public:

    OpenNaturalCubicSpline() = default;

    bool isValid() const;

    void setControlPoints( const std::vector< float >& controlPoints );

    int numControlPoints() const;
    float getControlPoint( int i ) const;
    void setControlPoint( int i, float p );
    void insertControlPoint( int i, float p );

    void appendControlPoint( float controlPoint );

    // output = x( t )
    // t between 0 and 1
    float evaluateAt( float t ) const;

    // output = dx/dt ( t )
    // t between 0 and 1
    float derivativeAt( float t ) const;

    // get the inverse spline
    // given a value of x, find t such that x(t) = x
    float inverse( float x, float tGuess, float epsilon = 0.001f,
                  int maxIterations = 50 );

private:

    void computeCoefficients();

    std::vector< float > m_vControlPoints;
    std::vector< Vector4f > m_vCoefficients;

};
