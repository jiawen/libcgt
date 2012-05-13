#pragma once

#include <QThread>
#include <imageproc/Image4ub.h>
#include <common/Array2D.h>

#include <Windows.h>
#include <NuiApi.h>
#include "QKinect.h"

class QKinectThread : public QThread
{
	Q_OBJECT

public:

	QKinectThread( std::shared_ptr< QKinect > pKinect, int pollingIntervalMS = 30 );

	int pollingIntervalMS() const;
	void setPollingIntervalMS( int pollingIntervalMS );

	void stop();

	std::shared_ptr< QKinect > kinect();

signals:

	void skeletonFrameReady( const NUI_SKELETON_FRAME& skeletonFrame );
	void colorFrameReady( const Image4ub& rgba );
	void depthFrameReady( const Array2D< ushort >& depth );

protected:

	virtual void run();

private:

	bool m_running;
	int m_pollingIntervalMS;
	std::shared_ptr< QKinect > m_pKinect;
};
