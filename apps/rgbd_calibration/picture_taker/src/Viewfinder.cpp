#include "Viewfinder.h"

#include <QBrush>
#include <QPainter>
#include <QTimer>

#include <core/common/ArrayUtils.h>
#include <core/imageproc/ColorMap.h>
#include <core/imageproc/Swizzle.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <qt_interop/qimage.h>
#include <third_party/pystring.h>

using libcgt::core::arrayutils::copy;
using libcgt::core::arrayutils::flipX;
using libcgt::core::imageproc::linearRemapToLuminance;
using libcgt::core::imageproc::RGBAToBGRA;
using libcgt::core::imageproc::RGBToBGRA;
using libcgt::camera_wrappers::kinect1x::KinectCamera;
using libcgt::camera_wrappers::openni2::OpenNI2Camera;
using libcgt::camera_wrappers::PixelFormat;
using libcgt::camera_wrappers::StreamConfig;
using libcgt::qt_interop::viewRGB32AsBGRX;
using libcgt::qt_interop::viewRGB32AsBGR;

#define USE_KINECT1X 1
#define USE_OPENNI2 0

Viewfinder::Viewfinder( const std::string& dir, QWidget* parent ) :
#if USE_KINECT1X
    m_kinect1xCamera( std::unique_ptr< KinectCamera >( new KinectCamera ) ),
    m_bgra( m_kinect1xCamera->colorResolution() ),
    m_infrared( m_kinect1xCamera->colorResolution() ),
    m_image( m_kinect1xCamera->colorResolution().x,
        m_kinect1xCamera->colorResolution().y,
        QImage::Format_RGB32 ),
#endif

#if USE_OPENNI2
    m_oniCamera( std::unique_ptr< OpenNI2Camera >( new OpenNI2Camera
        (
            StreamConfig( { 640, 480 }, 30, PixelFormat::RGB_U888, false ),
            StreamConfig(),
            StreamConfig()
        )
    ) ),
    m_rgb( m_oniCamera->colorConfig().resolution ),
    m_infrared( m_oniCamera->colorConfig().resolution ),
    m_image( m_oniCamera->colorConfig().resolution.x,
        m_oniCamera->colorConfig().resolution.y,
        QImage::Format_RGB32 ),
#endif
    m_colorNFB( pystring::os::path::join( dir, "color_" ), ".png" ),
    m_infraredNFB( pystring::os::path::join( dir, "ir_" ), ".png" ),
    QWidget( parent )
{
    if( m_kinect1xCamera != nullptr )
    {
        resize( m_kinect1xCamera->colorResolution().x,
            m_kinect1xCamera->colorResolution().y );
    }
    else
    {
        m_oniCamera->start();
        resize( m_oniCamera->colorConfig().resolution.x,
            m_oniCamera->colorConfig().resolution.y );
    }

    QTimer* viewfinderTimer = new QTimer( this );
    connect( viewfinderTimer, SIGNAL( timeout() ), this, SLOT( onViewfinderTimeout() ) );
    viewfinderTimer->setInterval( 0 );
    viewfinderTimer->start();

    QTimer* shotTimer = new QTimer( this );
    connect( shotTimer, SIGNAL( timeout() ), this, SLOT( onShotTimeout() ) );
    shotTimer->setInterval( 1000 );
    shotTimer->start();
}

void Viewfinder::updateRGB( Array2DView< const uint8x3 > frame )
{
    if( frame.width() != m_image.width() ||
        frame.height() != m_image.height() )
    {
        return;
    }

    auto dst = viewRGB32AsBGRX( m_image );
    RGBToBGRA( frame, dst ); // HACK: need RGBToBGRA
}

void Viewfinder::updateBGRA( Array2DView< const uint8x4 > frame )
{
    if( frame.width() != m_image.width() ||
        frame.height() != m_image.height() )
    {
        return;
    }

    auto dst = viewRGB32AsBGRX( m_image );
    copy( frame, dst );
}

void Viewfinder::updateInfrared( Array2DView< const uint16_t > frame )
{
    if( frame.width() != m_image.width() ||
        frame.height() != m_image.height() )
    {
        return;
    }

    auto dst = viewRGB32AsBGR( m_image );

    Range1i srcRange;
    if( m_kinect1xCamera != nullptr )
    {
        srcRange = Range1i::fromMinMax( 64, 65536 );
    }
    else
    {
        srcRange = Range1i::fromMinMax( 0, 1024 );
    }

    Range1i dstRange( 256 );
    linearRemapToLuminance( frame, srcRange, dstRange, dst );

    update();
}

// virtual
void Viewfinder::paintEvent( QPaintEvent* e )
{
    QPainter painter( this );

    painter.drawImage( 0, 0, m_image );

    painter.setPen( m_yellowPen );
    QFont font = painter.font();
    font.setPointSize( 96 );
    painter.setFont( font );
    int flags = Qt::AlignCenter;
    painter.drawText( 0, 0, 640, 480, flags,
        QString( "%1" ).arg( m_nSecondsUntilNextShot ) );

    if( m_nDrawFlashFrames > 0 )
    {
        painter.setBrush( m_whiteBrush );
        painter.drawRect( 0, 0, m_image.width(), m_image.height() );

        --m_nDrawFlashFrames;
    }
}

void Viewfinder::onViewfinderTimeout()
{
    if( m_kinect1xCamera != nullptr )
    {
        KinectCamera::Frame frame;
        frame.color = flipX( m_bgra.writeView() );
        frame.infrared = flipX( m_infrared.writeView() );
        if( m_kinect1xCamera->pollOne( frame ) )
        {
            if( m_isColor )
            {
                updateBGRA( flipX( frame.color ) );
                update();
            }
            else
            {
                updateInfrared( flipX( frame.infrared ) );
                update();
            }
        }
    }
    else if( m_oniCamera != nullptr )
    {
        OpenNI2Camera::Frame frame;
        frame.rgb = m_rgb.writeView();
        frame.infrared = m_infrared.writeView();
        if( m_oniCamera->pollOne( frame ) )
        {
            if( m_isColor )
            {
                updateRGB( frame.rgb );
                update();
            }
            else
            {
                updateInfrared( frame.infrared );
                update();
            }
        }
    }
}

void Viewfinder::onShotTimeout()
{
    --m_nSecondsUntilNextShot;
    if( m_nSecondsUntilNextShot == 0 )
    {
        // Take a shot and reset the camera.
        m_nDrawFlashFrames = kDefaultDrawFlashFrames;

#if 0
        if( m_isColor )
        {
            m_image.save( QString::fromStdString( m_colorNFB.filenameForNumber( m_nextColorImageIndex ) ) );
            ++m_nextColorImageIndex;
        }
        else
        {
            m_image.save( QString::fromStdString( m_infraredNFB.filenameForNumber( m_nextInfraredImageIndex ) ) );
            ++m_nextInfraredImageIndex;
        }
#endif

        // Toggle the next shot between color and infrared.
        m_isColor = !m_isColor;

        if( m_kinect1xCamera != nullptr )
        {
            m_kinect1xCamera.reset();

            StreamConfig colorConfig;
            StreamConfig depthConfig;
            StreamConfig infraredConfig;
            if( m_isColor )
            {
                colorConfig =
                    StreamConfig( { 640, 480 }, 30, PixelFormat::RGB_U888, false );
                m_nSecondsUntilNextShot = kColorShotIntervalSeconds;
            }
            else
            {
                infraredConfig =
                    StreamConfig{ { 640, 480 }, 30, PixelFormat::GRAY_U16, false };
                m_nSecondsUntilNextShot = kInfraredShotIntervalSeconds;
            }

            m_kinect1xCamera = std::unique_ptr< KinectCamera >
            (
                new KinectCamera( colorConfig, depthConfig, infraredConfig )
            );
        }
        else
        {
            m_oniCamera.reset();

            StreamConfig colorConfig;
            StreamConfig depthConfig;
            StreamConfig infraredConfig;
            if( m_isColor )
            {
                colorConfig =
                    StreamConfig( { 640, 480 }, 30, PixelFormat::RGB_U888, false );
                m_nSecondsUntilNextShot = kColorShotIntervalSeconds;
            }
            else
            {
                infraredConfig =
                    StreamConfig{ { 640, 480 }, 30, PixelFormat::GRAY_U16, false };
                m_nSecondsUntilNextShot = kInfraredShotIntervalSeconds;
            }

            m_oniCamera = std::unique_ptr< OpenNI2Camera >( new OpenNI2Camera(
                colorConfig, depthConfig, infraredConfig ) );
            m_oniCamera->start();
        }
    }
}
