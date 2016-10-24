#include "Viewfinder.h"

#include <QBrush>
#include <QPainter>
#include <QTimer>
#include <QThread>

#include <third_party/pystring/pystring.h>

#include <core/common/ArrayUtils.h>
#include <core/imageproc/ColorMap.h>
#include <core/imageproc/Swizzle.h>
#include <core/io/File.h>
#include <qt_interop/qimage.h>

using libcgt::core::arrayutils::copy;
using libcgt::core::imageproc::linearRemapToLuminance;
using libcgt::core::imageproc::RGBAToBGRA;
using libcgt::core::imageproc::RGBToBGRA;
using libcgt::camera_wrappers::openni2::OpenNI2Camera;
using libcgt::camera_wrappers::RGBDOutputStream;
using libcgt::camera_wrappers::PixelFormat;
using libcgt::camera_wrappers::StreamConfig;
using libcgt::camera_wrappers::StreamMetadata;
using libcgt::qt_interop::viewGrayscale8;
using libcgt::qt_interop::viewRGB32AsBGRX;
using libcgt::qt_interop::viewRGB32AsBGR;

Viewfinder::Viewfinder( const std::vector< StreamConfig >& streamConfig,
    const std::string& dir, QWidget* parent ) :
    m_nfb( pystring::os::path::join( dir, "recording_" ), ".rgbd" ),
    m_oniCamera( std::unique_ptr< OpenNI2Camera >(
        new OpenNI2Camera( streamConfig ) ) ),
    m_streamConfig( streamConfig ),
    m_rgb( m_oniCamera->colorConfig().resolution ),
    m_depth( m_oniCamera->depthConfig().resolution ),
    m_infrared( m_oniCamera->infraredConfig().resolution ),
    m_colorImage( m_oniCamera->colorConfig().resolution.x,
        m_oniCamera->colorConfig().resolution.y,
        QImage::Format_RGB32 ),
    m_depthImage( m_oniCamera->depthConfig().resolution.x,
        m_oniCamera->depthConfig().resolution.y,
        QImage::Format_Grayscale8 ),
    m_infraredImage( m_oniCamera->infraredConfig().resolution.x,
        m_oniCamera->infraredConfig().resolution.y,
        QImage::Format_Grayscale8 ),
    QWidget( parent )
{
    if( m_oniCamera->isValid() )
    {
        for( int i = 0; i < static_cast< int >( m_streamConfig.size() ); ++i )
        {
            const StreamConfig& config = m_streamConfig[ i ];
            m_outputMetadata.push_back( StreamMetadata{
                config.type, config.pixelFormat, config.resolution } );
            if( config.type == StreamType::COLOR )
            {
                m_colorStreamIndex = i;
            }
            if( config.type == StreamType::DEPTH )
            {
                m_depthStreamIndex = i;
            }
            if( config.type == StreamType::INFRARED )
            {
                m_infraredStreamIndex = i;
            }
        }

        m_oniCamera->start();
        setFixedSize
        (
            m_oniCamera->colorConfig().resolution.x +
            m_oniCamera->depthConfig().resolution.x +
            m_oniCamera->infraredConfig().resolution.x,

            std::max
            (
                {
                    m_oniCamera->colorConfig().resolution.y,
                    m_oniCamera->depthConfig().resolution.y,
                    m_oniCamera->infraredConfig().resolution.y
                }
            )
        );

        QTimer* viewfinderTimer = new QTimer( this );
        connect( viewfinderTimer, &QTimer::timeout,
            this, &Viewfinder::onViewfinderTimeout );
        viewfinderTimer->setInterval( 0 );
        viewfinderTimer->start();
    }
}

void Viewfinder::updateRGB( Array2DView< const uint8x3 > frame )
{
    if( frame.width() != m_colorImage.width() ||
        frame.height() != m_colorImage.height() )
    {
        return;
    }

    auto dst = viewRGB32AsBGRX( m_colorImage );
    RGBToBGRA( frame, dst );
}

void Viewfinder::updateBGRA( Array2DView< const uint8x4 > frame )
{
    if( frame.width() != m_colorImage.width() ||
        frame.height() != m_colorImage.height() )
    {
        return;
    }

    auto dst = viewRGB32AsBGRX( m_colorImage );
    copy( frame, dst );
}

void Viewfinder::updateDepth( Array2DView< const uint16_t > frame )
{
    if (frame.width() != m_depthImage.width() ||
        frame.height() != m_depthImage.height())
    {
        return;
    }

    auto dst = viewGrayscale8( m_depthImage );
    Range1i srcRange = Range1i::fromMinMax( 800, 4000 );
    Range1i dstRange = Range1i::fromMinMax( 51, 256 );
    linearRemapToLuminance( frame, srcRange, dstRange, dst );
}

void Viewfinder::updateInfrared( Array2DView< const uint16_t > frame )
{
    if (frame.width() != m_infraredImage.width() ||
        frame.height() != m_infraredImage.height())
    {
        return;
    }

    auto dst = viewGrayscale8( m_infraredImage );

    Range1i srcRange( 1024 );
    Range1i dstRange( 256 );
    linearRemapToLuminance( frame, srcRange, dstRange, dst );

    update();
}

void Viewfinder::setAeEnabled( bool enabled )
{
    bool b = m_oniCamera->setAutoExposureEnabled( enabled );
    printf( "Setting AE to: %d, result = %d\n", enabled, b );
}

void Viewfinder::setExposure( int value )
{
    printf( "Setting exposure to %d\n" , value);
    m_oniCamera->setExposure( value );
}

void Viewfinder::setGain( int value )
{
    printf( "Setting gain to %d\n", value );
    m_oniCamera->setGain( value );
}

void Viewfinder::setAwbEnabled( bool enabled )
{
    m_oniCamera->setAutoWhiteBalanceEnabled( enabled );
}

// virtual
void Viewfinder::paintEvent( QPaintEvent* e )
{
    QPainter painter( this );

    painter.drawImage( 0, 0, m_colorImage );
    painter.drawImage( m_colorImage.width(), 0, m_depthImage );
    painter.drawImage(
        m_colorImage.width() + m_depthImage.width(), 0, m_infraredImage );
}

void Viewfinder::startWriting()
{
    if( !m_outputStream.isValid() )
    {
        std::string filename = m_nfb.filenameForNumber( m_nextFileNumber );
        while( File::exists( filename.c_str() ) )
        {
            ++m_nextFileNumber;
            filename = m_nfb.filenameForNumber( m_nextFileNumber );
        }
        m_filename = filename;
        m_outputStream = RGBDOutputStream(
            m_outputMetadata, m_filename.c_str() );

        emit statusChanged( QString( "Writing to: " ) +
            QString::fromStdString( m_filename ) );
    }
}

void Viewfinder::stopWriting()
{
    m_filename = "";
    m_outputStream.close();
    emit statusChanged( QString("Idle.") );
}

void Viewfinder::onViewfinderTimeout()
{
    if( m_oniCamera != nullptr )
    {
        OpenNI2Camera::Frame frame;
        frame.rgb = m_rgb.writeView();
        frame.depth = m_depth.writeView();
        frame.infrared = m_infrared.writeView();

        bool pollResult = m_oniCamera->pollAll( frame, 33 );
        {
            if( frame.colorUpdated )
            {
                updateRGB( frame.rgb );
            }
            if( frame.depthUpdated )
            {
                updateDepth( frame.depth );
            }
            if( frame.infraredUpdated )
            {
                updateInfrared( frame.infrared );
            }

            if( m_outputStream.isValid() )
            {
                writeFrame( frame );
            }
            update();
        }
    }
}

void Viewfinder::writeFrame( OpenNI2Camera::Frame frame )
{
    QString status = QString( "Writing to: " ) +
        QString::fromStdString( m_filename );
    if( frame.colorUpdated )
    {
        status += QString( " color frame %1 " ).arg( frame.colorFrameNumber );
        Array1DView< uint8_t > view( frame.rgb,
            frame.rgb.numElements() * sizeof( uint8x3 ) );
        m_outputStream.write( m_colorStreamIndex, frame.colorFrameNumber,
            frame.colorTimestamp, view );
    }
    if( frame.depthUpdated )
    {
        status += QString( " depth frame %1 " ).arg( frame.depthFrameNumber );
        Array1DView< uint8_t > view( frame.depth,
            frame.depth.numElements() * sizeof( uint16_t ) );
        m_outputStream.write( m_depthStreamIndex, frame.depthFrameNumber,
            frame.depthTimestamp, view );
    }
    if( frame.infraredUpdated )
    {
        status += QString( " infrared frame %1 " ).arg(
            frame.infraredFrameNumber );
        Array1DView< uint8_t > view( frame.infrared,
            frame.infrared.numElements() * sizeof( uint16_t ) );
        m_outputStream.write( m_infraredStreamIndex, frame.infraredFrameNumber,
            frame.infraredTimestamp, view );
    }
    emit statusChanged( status );
}
