#include "DepthAveragerViewfinder.h"

#include <QBrush>
#include <QKeyEvent>
#include <QPainter>
#include <QTimer>

#include <core/common/ArrayUtils.h>
#include <core/common/Iterators.h>
#include <core/imageproc/ColorMap.h>
#include <core/imageproc/Swizzle.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <qt_interop/qimage.h>
#include <third_party/pystring.h>

DECLARE_int32( start_after );

using libcgt::core::imageproc::linearRemapToLuminance;
using libcgt::camera_wrappers::kinect1x::KinectCamera;
using libcgt::camera_wrappers::openni2::OpenNI2Camera;
using libcgt::camera_wrappers::PixelFormat;
using libcgt::camera_wrappers::StreamConfig;
using libcgt::qt_interop::viewGrayscale8;

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

DepthAveragerViewfinder::DepthAveragerViewfinder( const std::string& dir,
    QWidget* parent ) :
    m_isDryRun( dir == "" ),
    m_depthNFB( pystring::os::path::join( dir, "depth_average_" ), ".png" ),
    m_infraredNFB( pystring::os::path::join( dir, "infrared_average_" ),
        ".png" ),
    QWidget( parent )
{
    const auto& config = DEPTH_INFRARED_CONFIG;
    m_depth.resize( config[ 0 ].resolution );
    m_depthSum.resize( config[ 0 ].resolution );
    m_depthSum.fill( 0 );
    m_depthWeight.resize( config[ 0 ].resolution );
    m_depthWeight.fill( 0 );
    m_infrared.resize( config[ 1 ].resolution );
    m_infraredSum.resize( config[ 1 ].resolution );
    m_infraredSum.fill( 0 );

    m_oniCamera = std::unique_ptr< OpenNI2Camera >(
        new OpenNI2Camera( config ) );
    m_oniCamera->start();

    m_depthImage = QImage( m_depth.width(), m_depth.height(),
        QImage::Format_Grayscale8 );
    m_depthAverageImage = QImage( m_depth.width(), m_depth.height(),
        QImage::Format_Grayscale8 );
    m_infraredImage = QImage( m_infrared.width(), m_infrared.height(),
        QImage::Format_Grayscale8 );
    m_infraredAverageImage = QImage( m_infrared.width(), m_infrared.height(),
        QImage::Format_Grayscale8 );

    resize( m_depthImage.width() + m_infraredImage.width(),
        m_depthImage.height() + m_infraredImage.height() );

    QTimer* viewfinderTimer = new QTimer( this );
    connect( viewfinderTimer, &QTimer::timeout,
        this, &DepthAveragerViewfinder::onViewfinderTimeout );
    viewfinderTimer->setInterval( 0 );
    viewfinderTimer->start();
}

#include <math/MathUtils.h>

using libcgt::core::math::clamp;
using libcgt::core::math::rescale;

void DepthAveragerViewfinder::updateDepth( Array2DView< const uint16_t > frame )
{
    if( frame.width() != m_depthImage.width() ||
        frame.height() != m_depthImage.height() )
    {
        return;
    }

    auto dst = viewGrayscale8( m_depthImage );
    Range1i srcRange = Range1i::fromMinMax( 800, 4000 );
    Range1i dstRange = Range1i::fromMinMax( 51, 256 );
    linearRemapToLuminance( frame, srcRange, dstRange, dst );

    // Accumulate.
    Iterators::for2D( frame.size(),
        [&] ( int x, int y )
        {
            uint16_t z = frame[ {x, y} ];
            if( srcRange.contains( z ) )
            {
                m_depthSum[ {x, y} ] += z;
                m_depthWeight[ {x, y} ] += 1;
            }
        }
    );

    printf( "frame[320,240] = %d\n", frame[ {320, 240} ] );
    printf( "dst[320,240] = %d\n", dst[ {320, 240} ] );

    // Update average.
    auto dst2 = viewGrayscale8( m_depthAverageImage );
    Iterators::for2D( frame.size(),
        [&] ( int x, int y )
        {
            int32_t w = m_depthWeight[ {x, y} ];
            if( w > 0 )
            {
                uint64_t srcZ = m_depthSum[ {x, y} ];
                uint16_t div = srcZ / w;

                uint8_t tonemapped = static_cast< uint8_t >(
                    clamp( rescale( div, srcRange, dstRange ),
                    Range1i( 256 ) ) );

                uint16_t averageZ =
                    static_cast< uint16_t >(
                        m_depthSum[ {x, y} ] / static_cast< uint64_t >( w )
                    );
                dst2[ {x, y} ] = static_cast< uint8_t >(
                    clamp( rescale( averageZ, srcRange, dstRange ),
                    Range1i( 256 ) ) );
            }
            else
            {
                dst2[ {x, y} ] = 0;
            }
        }
    );

    printf( "sum[320,240] = %lld\n", m_depthSum[ {320, 240} ] );
    printf( "weight[320,240] = %d\n", m_depthWeight[ {320, 240} ] );
    printf( "dst2[320,240] = %d\n", dst2[ {320, 240} ] );
}

void DepthAveragerViewfinder::updateInfrared( Array2DView< const uint16_t > frame )
{
    if( frame.width() != m_infraredImage.width() ||
        frame.height() != m_infraredImage.height() )
    {
        return;
    }

    auto dst = viewGrayscale8( m_infraredImage );

#if 0
    Range1i srcRange;
    if( m_kinect1xCamera != nullptr )
    {
        srcRange = Range1i::fromMinMax( 64, 65536 );
    }
    else
    {
        srcRange = Range1i::fromMinMax( 0, 1024 );
    }
#endif

    Range1i srcRange = Range1i::fromMinMax( 0, 1024 );

    Range1i dstRange( 256 );
    linearRemapToLuminance( frame, srcRange, dstRange, dst );

    // Accumulate.
    Iterators::for2D( frame.size(),
        [&] ( int x, int y )
        {
            m_infraredSum[ {x, y} ] += frame[ {x, y} ];
        }
    );
    ++m_infraredWeight;

    // Update average.
    auto dst2 = viewGrayscale8( m_infraredAverageImage );
    Iterators::for2D( frame.size(),
        [&] ( int x, int y )
        {
            uint16_t averageV = m_infraredSum[ {x, y} ] / m_infraredWeight;
            dst2[ {x, y} ] = static_cast< uint8_t >(
                clamp( rescale( averageV, srcRange, dstRange ),
                Range1i( 256 ) ) );
        }
    );
}

// virtual
void DepthAveragerViewfinder::paintEvent( QPaintEvent* e )
{
    QPainter painter( this );

    painter.drawImage( 0, 0, m_depthImage );
    painter.drawImage( m_depthImage.width(), 0, m_infraredImage );
    painter.drawImage( 0, m_depthImage.height(), m_depthAverageImage );
    painter.drawImage( m_depthImage.width(), m_depthImage.height(),
        m_infraredAverageImage );

    // TODO(jiawen): countdown to start then flash
#if 0
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
#endif
}

// virtual
void DepthAveragerViewfinder::keyPressEvent( QKeyEvent* e )
{
    if( e->key() == Qt::Key_D )
    {
        saveDepthAverage();
    }
    else if( e->key() == Qt::Key_F )
    {
        resetDepthAverage();
    }
    else if( e->key() == Qt::Key_I )
    {
        saveInfraredAverage();
    }
    else if( e->key() == Qt::Key_O )
    {
        resetInfraredAverage();
    }
}

void DepthAveragerViewfinder::onViewfinderTimeout()
{
#if 0
    if( m_kinect1xCamera != nullptr )
    {

    }
#endif

    if( m_oniCamera != nullptr )
    {
        OpenNI2Camera::Frame frame;
        frame.depth = m_depth.writeView();
        frame.infrared = m_infrared.writeView();
        if( m_oniCamera->pollOne( frame ) )
        {
            if( frame.depthUpdated )
            {
                updateDepth( frame.depth );
            }
            if( frame.infraredUpdated )
            {
                updateInfrared( frame.infrared );
            }
            if( frame.depthUpdated || frame.infraredUpdated )
            {
                update();
            }
        }
    }
}

void DepthAveragerViewfinder::resetDepthAverage()
{
    m_depthSum.fill( 0 );
    m_depthWeight.fill( 0 );
    m_depthAverageImage.fill( 0 );
    update();
}

void DepthAveragerViewfinder::saveDepthAverage()
{
    if( !m_isDryRun )
    {
        m_depthAverageImage.save( QString::fromStdString(
            m_depthNFB.filenameForNumber( m_nextDepthAverageImageIndex ) ) );
        ++m_nextDepthAverageImageIndex;
    }
}

void DepthAveragerViewfinder::resetInfraredAverage()
{
    m_infraredSum.fill( 0 );
    m_infraredWeight = 0;
    m_infraredAverageImage.fill( 0 );
    update();
}

void DepthAveragerViewfinder::saveInfraredAverage()
{
    if( !m_isDryRun )
    {
        m_infraredAverageImage.save( QString::fromStdString(
            m_infraredNFB.filenameForNumber(
                m_nextInfraredAverageImageIndex ) ) );
        ++m_nextInfraredAverageImageIndex;
    }
}
