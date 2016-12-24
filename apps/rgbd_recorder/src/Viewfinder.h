#include <memory>

#include <QBrush>
#include <QPen>
#include <QWidget>

#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/common/Array2D.h"
#include "libcgt/core/io/NumberedFilenameBuilder.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/camera_wrappers/RGBDStream.h"
#include "libcgt/camera_wrappers/OpenNI2/OpenNI2Camera.h"

class Viewfinder : public QWidget
{
    Q_OBJECT

public:

    using OpenNI2Camera = libcgt::camera_wrappers::openni2::OpenNI2Camera;
    using StreamConfig = libcgt::camera_wrappers::StreamConfig;
    using StreamMetadata = libcgt::camera_wrappers::StreamMetadata;

    // dir should be something like "/tmp". The trailing slash is optional.
    Viewfinder( const std::vector< StreamConfig >& streamConfig,
        const std::string& dir, QWidget* parent = nullptr );

    void updateRGB( Array2DReadView< uint8x3 > frame );
    void updateBGRA( Array2DReadView< uint8x4 > frame );
    void updateDepth( Array2DReadView< uint16_t > frame );
    void updateInfrared( Array2DReadView< uint16_t > frame );

signals:

    void statusChanged( QString str );

public slots:

    void setAeEnabled( bool enabled );
    void setExposure( int value );
    void setGain( int value );

    void setAwbEnabled( bool enabled );

    // TODO(jiawen): write metadata as well in a proto.
    void startWriting();
    void stopWriting();

protected:

    virtual void paintEvent( QPaintEvent* e ) override;

private:

    int m_nextFileNumber = 0;
    NumberedFilenameBuilder m_nfb;
    std::string m_filename;

    std::unique_ptr< OpenNI2Camera > m_oniCamera;
    std::vector< StreamConfig > m_streamConfig;
    std::vector< StreamMetadata > m_outputMetadata;
    int m_colorStreamIndex = -1;
    int m_depthStreamIndex = -1;
    int m_infraredStreamIndex = -1;

    Array2D< uint8x3 > m_rgb;
    Array2D< uint8x4 > m_bgra;
    Array2D< uint16_t > m_depth;
    Array2D< uint16_t > m_infrared;

    QImage m_colorImage;
    QImage m_depthImage;
    QImage m_infraredImage;

    const QPen m_yellowPen = QPen{ Qt::yellow };
    const QBrush m_whiteBrush = QBrush{ Qt::white };

    void writeFrame( OpenNI2Camera::FrameView frame );

    libcgt::camera_wrappers::RGBDOutputStream m_outputStream;

private slots:

    void onViewfinderTimeout();
};
