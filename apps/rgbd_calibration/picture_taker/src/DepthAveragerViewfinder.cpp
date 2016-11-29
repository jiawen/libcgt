#include "DepthAveragerViewfinder.h"

#include <QBrush>
#include <QKeyEvent>
#include <QPainter>
#include <QTimer>

#include <third_party/pystring/pystring.h>

#include <core/common/ArrayUtils.h>
#include <core/common/ForND.h>
#include <core/geometry/RangeUtils.h>
#include <core/imageproc/ColorMap.h>
#include <core/imageproc/Swizzle.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <core/io/PortableFloatMapIO.h>
#include <core/math/MathUtils.h>
#include <qt_interop/qimage.h>

DECLARE_int32( start_after );

using libcgt::core::for2D;
using libcgt::core::imageproc::linearRemapToLuminance;
using libcgt::core::geometry::rescale;
using libcgt::core::math::clamp;
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
    m_depthPNGNFB( pystring::os::path::join( dir, "depth_average_" ), ".png" ),
    m_depthPFMNFB( pystring::os::path::join( dir, "depth_average_" ), ".pfm" ),
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
    m_depthAverage.resize( config[ 0 ].resolution );
    m_depthAverage.fill( 0 );
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

void DepthAveragerViewfinder::updateDepth( Array2DReadView< uint16_t > frame )
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
    for2D( frame.size(),
        [&] ( const Vector2i& xy )
        {
            uint16_t z = frame[ xy ];
            if( srcRange.contains( z ) )
            {
                m_depthSum[ xy ] += z;
                m_depthWeight[ xy ] += 1;
            }
        }
    );

    // Update average.
    auto averageImageView = viewGrayscale8( m_depthAverageImage );
    for2D( frame.size(),
        [&] ( const Vector2i& xy )
        {
            int32_t w = m_depthWeight[ xy ];
            if( w > 0 )
            {
                uint64_t srcZ = m_depthSum[ xy ];
                uint16_t q16 = srcZ / w;

                uint8_t tonemapped = static_cast< uint8_t >(
                    clamp( rescale( q16, srcRange, dstRange ),
                    Range1i( 256 ) ) );

                uint16_t averageZ =
                    static_cast< uint16_t >(
                        m_depthSum[ xy ] / static_cast< uint64_t >( w )
                    );
                averageImageView[ xy ] = static_cast< uint8_t >(
                    clamp( rescale( averageZ, srcRange, dstRange ),
                    Range1i( 256 ) ) );

                // Carefully divide to get float:
                // x div y --> (q, r) such that x = q * y + r. r in [0, y).
                // A nice floating point quotient would be:
                // float(q) + float(r)/y.
                std::lldiv_t qr = std::lldiv( srcZ, w );
                float q = static_cast< float >( qr.quot ) +
                    static_cast< float >( qr.rem ) / w;
                m_depthAverage[ xy ] = q;
            }
            else
            {
                averageImageView[ xy ] = 0;
                m_depthAverage[ xy ] = 0;
            }
        }
    );
}

void DepthAveragerViewfinder::updateInfrared(
    Array2DReadView< uint16_t > frame )
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
    for2D( frame.size(),
        [&] ( const Vector2i& xy )
        {
            m_infraredSum[ xy ] += frame[ xy ];
        }
    );
    ++m_infraredWeight;

    // Update average.
    auto dst2 = viewGrayscale8( m_infraredAverageImage );
    for2D( frame.size(),
        [&] ( const Vector2i& xy )
        {
            uint16_t averageV = m_infraredSum[ xy ] / m_infraredWeight;
            dst2[ xy ] = static_cast< uint8_t >(
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
        OpenNI2Camera::FrameView frame;
        frame.depth = m_depth;
        frame.infrared = m_infrared;
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
    m_depthAverage.fill( 0 );
    update();
}

void DepthAveragerViewfinder::saveDepthAverage()
{
    if( !m_isDryRun )
    {
        m_depthAverageImage.save( QString::fromStdString(
            m_depthPNGNFB.filenameForNumber(
                m_nextDepthAverageImageIndex ) ) );

        std::string pfmFilename = m_depthPFMNFB.filenameForNumber(
            m_nextDepthAverageImageIndex );
        PortableFloatMapIO::write( pfmFilename, m_depthAverage );

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
