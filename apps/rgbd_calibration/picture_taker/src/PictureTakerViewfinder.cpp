#include "PictureTakerViewfinder.h"

#include <QBrush>
#include <QKeyEvent>
#include <QPainter>
#include <QTimer>

#include <third_party/pystring/pystring.h>

#include "libcgt/core/common/ArrayUtils.h"
#include "libcgt/core/imageproc/ColorMap.h"
#include "libcgt/core/imageproc/Swizzle.h"
#include "libcgt/core/io/NumberedFilenameBuilder.h"
#include "libcgt/qt_interop/qimage.h"

DECLARE_int32( start_after );
DECLARE_string( mode );
DECLARE_int32( shot_interval );
DECLARE_int32( secondary_shot_interval );

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
    StreamConfig{StreamType::COLOR, { 640, 480 }, PixelFormat::RGB_U888,
        30, false },
    StreamConfig{ StreamType::DEPTH, { 640, 480 }, PixelFormat::DEPTH_MM_U16,
        30, false },
};

const std::vector< StreamConfig > DEPTH_INFRARED_CONFIG =
{
    StreamConfig{ StreamType::DEPTH, { 640, 480 }, PixelFormat::DEPTH_MM_U16,
        30, false },
    StreamConfig{ StreamType::INFRARED, { 640, 480 }, PixelFormat::GRAY_U16,
        30, false }
};

PictureTakerViewfinder::PictureTakerViewfinder( const std::string& dir,
    QWidget* parent ) :
    m_isDryRun( dir == "" ),
    m_colorNFB( pystring::os::path::join( dir, "color_" ), ".png" ),
    m_infraredNFB( pystring::os::path::join( dir, "infrared_" ), ".png" ),
    m_nSecondsUntilNextShot( FLAGS_shot_interval ),
    QWidget( parent )
{
    const std::vector< StreamConfig >* config;
    if( FLAGS_mode == "color" )
    {
        m_isColor = true;
        config = &COLOR_ONLY_CONFIG;
        m_rgb.resize( config->at( 0 ).resolution );
    }
    else if( FLAGS_mode == "infrared" )
    {
        m_isColor = false;
        config = &INFRARED_ONLY_CONFIG;
        m_infrared.resize( config->at( 0 ).resolution );
    }
    else
    {
        // Start with color.
        m_isColor = true;
        config = &COLOR_ONLY_CONFIG;
        m_rgb.resize( COLOR_ONLY_CONFIG[ 0 ].resolution );
        m_infrared.resize( INFRARED_ONLY_CONFIG[ 0 ].resolution );
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

    m_image = QImage(
        static_cast< int >( std::max( m_rgb.width(), m_infrared.width() ) ),
        static_cast< int >( std::max( m_rgb.height(), m_infrared.height() ) ),
        QImage::Format_RGB32 );
#endif

    resize( m_image.width(), m_image.height() );

    if( FLAGS_start_after == 0 )
    {
        m_saving = true;
    }
    else
    {
        m_saving = false;
        QTimer::singleShot( FLAGS_start_after * 1000,
            [&] ()
            {
                m_saving = true;
            }
        );
    }

    QTimer* viewfinderTimer = new QTimer( this );
    connect( viewfinderTimer, &QTimer::timeout,
        this, &PictureTakerViewfinder::onViewfinderTimeout );
    viewfinderTimer->setInterval( 0 );
    viewfinderTimer->start();

    if( FLAGS_shot_interval > 0 )
    {
        QTimer* shotTimer = new QTimer( this );
        connect( shotTimer, &QTimer::timeout,
            this, &PictureTakerViewfinder::onShotTimeout );
        shotTimer->setInterval( 1000 );
        shotTimer->start();
    }
}

void PictureTakerViewfinder::updateRGB( Array2DReadView< uint8x3 > frame )
{
    if( frame.width() != m_image.width() ||
        frame.height() != m_image.height() )
    {
        return;
    }

    auto dst = viewRGB32AsBGRX( m_image );
    RGBToBGRA( frame, dst ); // HACK: need RGBToBGRA
}

void PictureTakerViewfinder::updateBGRA( Array2DReadView< uint8x4 > frame )
{
    if( frame.width() != m_image.width() ||
        frame.height() != m_image.height() )
    {
        return;
    }

    auto dst = viewRGB32AsBGRX( m_image );
    copy( frame, dst );
}

void PictureTakerViewfinder::updateInfrared(
    Array2DReadView< uint16_t > frame )
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
}

// virtual
void PictureTakerViewfinder::paintEvent( QPaintEvent* e )
{
    QPainter painter( this );

    painter.drawImage( 0, 0, m_image );

    if( FLAGS_shot_interval > 0 )
    {
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
}

// virtual
void PictureTakerViewfinder::keyPressEvent( QKeyEvent* e )
{
    if( e->key() == Qt::Key_Space )
    {
        maybeSaveShot();
        maybeToggleStreams();
    }
}

void PictureTakerViewfinder::onViewfinderTimeout()
{
    if( m_kinect1xCamera != nullptr )
    {
        KinectCamera::FrameView frame;
        frame.color = flipX< uint8x4 >( m_bgra );
        frame.infrared = flipX< uint16_t >( m_infrared );
        if( m_kinect1xCamera->pollOne( frame ) )
        {
            if( m_isColor )
            {
                // TODO: HACK: why do I have to flip it twice?
                updateBGRA( flipX< uint8x4 >( frame.color ) );
                update();
            }
            else
            {
                updateInfrared( flipX< uint16_t >( frame.infrared ) );
                update();
            }
        }
    }
    else if( m_oniCamera != nullptr )
    {
        OpenNI2Camera::FrameView frame;
        frame.color = m_rgb;
        frame.infrared = m_infrared;
        if( m_oniCamera->pollOne( frame ) )
        {
            if( m_isColor )
            {
                updateRGB( frame.color );
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

void PictureTakerViewfinder::onShotTimeout()
{
    --m_nSecondsUntilNextShot;
    if( m_nSecondsUntilNextShot == 0 )
    {
        // Take a shot and reset the camera.
        m_nDrawFlashFrames = kDefaultDrawFlashFrames;

        maybeSaveShot();
        maybeToggleStreams();
    }
}

void PictureTakerViewfinder::maybeSaveShot()
{
    if( !m_isDryRun && m_saving )
    {
        std::string filename;
        if( m_isColor )
        {
            filename = m_colorNFB.filenameForNumber( m_nextColorImageIndex );
            ++m_nextColorImageIndex;
        }
        else
        {
            filename = m_infraredNFB.filenameForNumber(
                m_nextInfraredImageIndex );
            ++m_nextInfraredImageIndex;
        }

        printf( "Saving: %s...", filename.c_str() );
        bool succeeded = m_image.save( QString::fromStdString( filename ) );
        if( succeeded )
        {
            printf( "ok.\n" );
        }
        else
        {
            printf( "FAILED.\n" );
        }
    }
}

void PictureTakerViewfinder::maybeToggleStreams()
{
    // If in toggle mode, do the toggling.
    if( FLAGS_mode == "color" )
    {
        m_nSecondsUntilNextShot = FLAGS_shot_interval;
    }
    else if( FLAGS_mode == "infrared" )
    {
        m_nSecondsUntilNextShot = FLAGS_shot_interval;
    }
    else if( FLAGS_mode == "toggleColorInfrared" )
    {
        // Toggle the next shot between color and infrared.
        m_isColor = !m_isColor;

        const std::vector< StreamConfig >* config;
        if( m_isColor )
        {
            config = &COLOR_ONLY_CONFIG;
            m_nSecondsUntilNextShot = FLAGS_shot_interval;
        }
        else
        {
            config = &INFRARED_ONLY_CONFIG;
            m_nSecondsUntilNextShot = FLAGS_secondary_shot_interval;
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
