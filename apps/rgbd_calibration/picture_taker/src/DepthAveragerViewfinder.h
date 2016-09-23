#include <memory>

#include <gflags/gflags.h>

#include <QBrush>
#include <QPen>
#include <QWidget>

#include <camera_wrappers/kinect1x/KinectCamera.h>
#include <camera_wrappers/OpenNI2/OpenNI2Camera.h>

#include <core/common/BasicTypes.h>
#include <core/common/Array2DView.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <core/vecmath/Vector2i.h>

class DepthAveragerViewfinder : public QWidget
{
    Q_OBJECT

public:

    using KinectCamera = libcgt::camera_wrappers::kinect1x::KinectCamera;
    using OpenNI2Camera = libcgt::camera_wrappers::openni2::OpenNI2Camera;

    // dir should be something like "/tmp". The trailing slash is optional.
    DepthAveragerViewfinder( const std::string& dir = "",
        QWidget* parent = nullptr );

    void updateDepth( Array2DView< const uint16_t > frame );
    void updateInfrared( Array2DView< const uint16_t > frame );

protected:

    virtual void paintEvent( QPaintEvent* e ) override;
    virtual void keyPressEvent( QKeyEvent* e ) override;

private:

    const bool m_isDryRun;

    std::unique_ptr< OpenNI2Camera > m_oniCamera;

    Array2D< uint16_t > m_depth;
    Array2D< uint64_t > m_depthSum;
    Array2D< int32_t > m_depthWeight;

    Array2D< uint16_t > m_infrared;
    Array2D< uint64_t > m_infraredSum;
    int32_t m_infraredWeight = 0;

    QImage m_depthImage;
    QImage m_depthAverageImage;
    QImage m_infraredImage;
    QImage m_infraredAverageImage;

    const QPen m_yellowPen = QPen{ Qt::yellow };
    const QBrush m_whiteBrush = QBrush{ Qt::white };

    void resetDepthAverage();
    void saveDepthAverage();

    void resetInfraredAverage();
    void saveInfraredAverage();

    NumberedFilenameBuilder m_depthNFB;
    int m_nextDepthAverageImageIndex = 0;
    NumberedFilenameBuilder m_infraredNFB;
    int m_nextInfraredAverageImageIndex = 0;

private slots:

    void onViewfinderTimeout();

};
