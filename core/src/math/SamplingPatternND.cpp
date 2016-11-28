#include "math/SamplingPatternND.h"

SamplingPatternND::SamplingPatternND( int nSamples, int nDimensions ) :
    m_nSamples( nSamples ),
    m_nDimensions( nDimensions ),
    m_samples( nSamples * nDimensions )
{
}

void SamplingPatternND::getSample( int j, Array1DWriteView< float > sample )
{
    int sampleStartIndex = j * m_nDimensions;
    for( int i = 0; i < m_nDimensions; ++i )
    {
        sample[i] = m_samples[ sampleStartIndex + i ];
    }
}

int SamplingPatternND::getNumSamples() const
{
    return m_nSamples;
}

int SamplingPatternND::getNumDimensions() const
{
    return m_nDimensions;
}

Array1DWriteView< float > SamplingPatternND::rawSamples()
{
    return m_samples;
}
