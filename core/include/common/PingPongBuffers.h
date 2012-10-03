#pragma once

template< typename T >
class PingPongBuffers
{
public:

	PingPongBuffers();

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
PingPongBuffers< T >::PingPongBuffers() :

	m_outputBufferIndex( 0 )

{

}

template< typename T >
const T& PingPongBuffers< T >::inputBuffer() const
{
	return m_buffers[ ( m_outputBufferIndex + 1 ) % 2 ];
}

template< typename T >
T& PingPongBuffers< T >::inputBuffer()
{
	return m_buffers[ ( m_outputBufferIndex + 1 ) % 2 ];
}

template< typename T >
const T& PingPongBuffers< T >::outputBuffer() const
{
	return m_buffers[ m_outputBufferIndex ];
}

template< typename T >
T& PingPongBuffers< T >::outputBuffer()
{
	return m_buffers[ m_outputBufferIndex ];
}

template< typename T >
void PingPongBuffers< T >::swapBuffers()
{
	m_outputBufferIndex = ( m_outputBufferIndex + 1 ) % 2;
}
