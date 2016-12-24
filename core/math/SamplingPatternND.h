#pragma once

#include "libcgt/core/common/Array1D.h"

class SamplingPatternND
{
public:

    SamplingPatternND( int nSamples, int nDimensions );

    // get the jth sample
    // sample[] should have length of at least nDimensions
    void getSample( int j, Array1DWriteView< float > sample );

    int getNumSamples() const;
    int getNumDimensions() const;

    // returns the raw data
    // to be populated by a sampling algorithm
    // afSamples is stored as:
    // if nDimensions = 5
    // mem: 01234 56780 ABCDE
    //      00000 11111 22222
    //      s0    s1    s2
    Array1DWriteView< float > rawSamples();

private:

    int m_nSamples;
    int m_nDimensions;
    Array1D< float > m_samples;
};
