#pragma once

template< typename T >
class PingPong
{
public:

	PingPong();

	const T& inputBuffer() const;
	T& inputBuffer();

	const T& outputBuffer() const;
	T& outputBuffer();

	void swapBuffers();

private:

	int m_outputBufferIndex;
	T m_buffers[2];
};

template< typename T >
PingPong< T >::PingPong() :

	m_outputBufferIndex( 0 )

{

}

template< typename T >
const T& PingPong< T >::inputBuffer() const
{
	return m_buffers[ ( m_outputBufferIndex + 1 ) % 2 ];
}

template< typename T >
T& PingPong< T >::inputBuffer()
{
	return m_buffers[ ( m_outputBufferIndex + 1 ) % 2 ];
}

template< typename T >
const T& PingPong< T >::outputBuffer() const
{
	return m_buffers[ m_outputBufferIndex ];
}

template< typename T >
T& PingPong< T >::outputBuffer()
{
	return m_buffers[ m_outputBufferIndex ];
}

template< typename T >
void PingPong< T >::swapBuffers()
{
	m_outputBufferIndex = ( m_outputBufferIndex + 1 ) % 2;
}
