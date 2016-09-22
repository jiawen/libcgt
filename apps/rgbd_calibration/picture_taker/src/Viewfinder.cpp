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

#define USE_KINECT1X 0
#define USE_OPENNI2 1

const std::vector< StreamConfig > COLOR_ONLY_CONFIG =
{
    StreamConfig
    {
        StreamType::COLOR, { 640, 480 }, PixelFormat::RGB_U888, 30, false
    }
};

const std::vector< StreamConfig > DEPTH_ONLY_CONFIG =
{
    StreamConfig
    {
        StreamType::DEPTH, { 640, 480 }, PixelFormat::DEPTH_MM_U16, 30, false
    }
};

const std::vector< StreamConfig > INFRARED_ONLY_CONFIG =
{
    StreamConfig
    {
        StreamType::INFRARED, { 640, 480 }, PixelFormat::GRAY_U16, 30, false
    }
};

const std::vector< StreamConfig > COLOR_DEPTH_CONFIG =
{
    StreamConfig{ StreamType::COLOR, { 640, 480 }, PixelFormat::RGB_U888, 30, false },
    StreamConfig{ StreamType::DEPTH, { 640, 480 }, PixelFormat::DEPTH_MM_U16, 30, false },
};

const std::vector< StreamConfig > DEPTH_INFRARED_CONFIG =
{
    StreamConfig{ StreamType::DEPTH, { 640, 480 }, PixelFormat::DEPTH_MM_U16, 30, false },
    StreamConfig{ StreamType::INFRARED, { 640, 480 }, PixelFormat::GRAY_U16, 30, false }
};

Viewfinder::Viewfinder( int mode, const std::string& dir, QWidget* parent ) :
    m_mode( mode ),
    m_isDryRun( dir == "" ),
    m_colorNFB( pystring::os::path::join( dir, "color_" ), ".png" ),
    m_infraredNFB( pystring::os::path::join( dir, "ir_" ), ".png" ),
    QWidget( parent )
{
    const std::vector< StreamConfig >* config = &COLOR_ONLY_CONFIG;

    if( mode == 0 || mode == 2 )
    {
        m_isColor = true;
    }
    else
    {
        m_isColor = false;
        config = &INFRARED_ONLY_CONFIG;
    }


#if USE_KINECT1X
    m_kinect1xCamera = std::unique_ptr< KinectCamera >(
        new KinectCamera( *config ) );

    m_kinect1xCamera( std::unique_ptr< KinectCamera >( new KinectCamera ) ),
    m_bgra.resize( m_kinect1xCamera->colorResolution() ),
    m_infrared.resize( m_kinect1xCamera->colorResolution() ),
    m_image.resize( m_kinect1xCamera->colorResolution().x,
        m_kinect1xCamera->colorResolution().y,
        QImage::Format_RGB32 )
#endif

#if USE_OPENNI2
    m_oniCamera = std::unique_ptr< OpenNI2Camera >(
        new OpenNI2Camera( *config ) );
    m_oniCamera->start();

    m_rgb.resize( m_oniCamera->colorConfig().resolution );
    m_infrared.resize( m_oniCamera->infraredConfig().resolution );
    m_image = QImage(
        std::max( m_rgb.width(), m_infrared.width() ),
        std::max( m_rgb.height(), m_infrared.height() ),
        QImage::Format_RGB32 );
#endif

    resize( m_image.width(), m_image.height() );

    QTimer::singleShot( kStartWaitTimeSeconds * 1000,
        [&]()
        {
            m_startSaving = true;
        }
    );

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

        if( !m_isDryRun && m_startSaving )
        {
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
        }

        // If in toggle mode, do the toggling.
        if( m_mode == 0 )
        {
            m_nSecondsUntilNextShot = kColorShotIntervalSeconds;
        }
        else if( m_mode == 1 )
        {
            m_nSecondsUntilNextShot = kInfraredShotIntervalSeconds;
        }
        if( m_mode == 2 )
        {
            // Toggle the next shot between color and infrared.
            m_isColor = !m_isColor;

            const std::vector< StreamConfig >* config;
            if( m_isColor )
            {
                config = &COLOR_ONLY_CONFIG;
                m_nSecondsUntilNextShot = kColorShotIntervalSeconds;
            }
            else
            {
                config = &INFRARED_ONLY_CONFIG;
                m_nSecondsUntilNextShot = kInfraredShotIntervalSeconds;
            }

            if( m_kinect1xCamera != nullptr )
            {
                m_kinect1xCamera.reset();
                m_kinect1xCamera = std::unique_ptr< KinectCamera >(
                    new KinectCamera( *config ) );
            }
            else
            {
                m_oniCamera.reset();
                m_oniCamera = std::unique_ptr< OpenNI2Camera >(
                    new OpenNI2Camera( *config ) );
                m_oniCamera->start();
            }
        }
    }
}
