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

class PictureTakerViewfinder : public QWidget
{
    Q_OBJECT

public:

    using KinectCamera = libcgt::camera_wrappers::kinect1x::KinectCamera;
    using OpenNI2Camera = libcgt::camera_wrappers::openni2::OpenNI2Camera;

    // dir should be something like "/tmp". The trailing slash is optional.
    PictureTakerViewfinder( const std::string& dir = "",
        QWidget* parent = nullptr );

    void updateRGB( Array2DView< const uint8x3 > frame );
    void updateBGRA( Array2DView< const uint8x4 > frame );
    void updateInfrared( Array2DView< const uint16_t > frame );

protected:

    virtual void paintEvent( QPaintEvent* e ) override;
    virtual void keyPressEvent( QKeyEvent* e ) override;

private:

    NumberedFilenameBuilder m_colorNFB;
    NumberedFilenameBuilder m_infraredNFB;

    const bool m_isDryRun;
    bool m_saving = false;

    bool m_isColor = true;
    std::unique_ptr< KinectCamera > m_kinect1xCamera;
    std::unique_ptr< OpenNI2Camera > m_oniCamera;

    Array2D< uint8x3 > m_rgb;
    Array2D< uint8x4 > m_bgra;
    Array2D< uint16_t > m_infrared;

    int m_nSecondsUntilNextShot;

    const int kDefaultDrawFlashFrames = 5;
    int m_nDrawFlashFrames = 0;

    QImage m_image;

    const QPen m_yellowPen = QPen{ Qt::yellow };
    const QBrush m_whiteBrush = QBrush{ Qt::white };

    int m_nextColorImageIndex = 0;
    int m_nextInfraredImageIndex = 0;

    void maybeSaveShot();
    void maybeToggleStreams();

private slots:

    void onViewfinderTimeout();
    void onShotTimeout();

};
