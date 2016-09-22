#include <QApplication>

#include <QGridLayout>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QSlider>

#include "Viewfinder.h"

#include <core/vecmath/Box3f.h>
#include <camera_wrappers/StreamConfig.h>

using libcgt::camera_wrappers::PixelFormat;
using libcgt::camera_wrappers::StreamConfig;

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

// Color and infrared disallowed.

int main( int argc, char* argv[] )
{
#if 0
    Box3f b({ 2, 2, 2 });

    Vector3f o{ 0, 0, -1.0f };
    Vector3f d{ 0, 0, 1 };

    float tNear;
    float tFar;
    bool isect = b.intersectLine(o, d, tNear, tFar);

    return 0;
#endif

    // TODO(jiawen): --ir, --out foo.rgbd or --outdir

    if( argc < 2 )
    {
        printf( "Usage: %s <dir>\n", argv[0] );
        printf( "Files will be saved to <dir>/recording_#####.rgbd" );
        return 1;
    }

    QApplication app( argc, argv );

    Viewfinder* vf = new Viewfinder( DEPTH_INFRARED_CONFIG, argv[ 1 ] );

    QGridLayout* layout = new QGridLayout;
    layout->addWidget( vf, 0, 0, 3, 1 );

    QCheckBox* aeCheckbox = new QCheckBox( "Enable AE" );
    aeCheckbox->setCheckState( Qt::CheckState::Checked );
    layout->addWidget( aeCheckbox, 0, 1 );
    QObject::connect( aeCheckbox, &QCheckBox::toggled,
        vf, &Viewfinder::setAeEnabled );

    // TODO(jiawen): Xtion Pro Live exposure range seems to be [0, 255]
    QSlider* exposureSlider = new QSlider( Qt::Orientation::Horizontal );
    exposureSlider->setMinimum( 0 );
    exposureSlider->setMaximum( 255 );
    exposureSlider->setValue( 128 );
    layout->addWidget( exposureSlider, 1, 1 );
    QObject::connect( exposureSlider, &QSlider::valueChanged,
        vf, &Viewfinder::setExposure );

    // TODO(jiawen): Xtion Pro Live does not seem to support gain.
#if 0
    QSlider* gainSlider = new QSlider( Qt::Orientation::Horizontal );
    gainSlider->setMinimum( 0 );
    gainSlider->setMaximum( 100 );
    layout->addWidget( gainSlider, 2, 1 );
    QObject::connect( gainSlider, &QSlider::valueChanged,
        vf, &Viewfinder::setGain );
#endif

    QCheckBox* awbCheckbox = new QCheckBox( "Enable AWB" );
    awbCheckbox->setCheckState( Qt::CheckState::Checked );
    layout->addWidget( awbCheckbox, 2, 1 );
    QObject::connect( awbCheckbox, &QPushButton::toggled,
        vf, &Viewfinder::setAwbEnabled );

    QPushButton* startRecordingButton = new QPushButton( "Start Recording" );
    layout->addWidget( startRecordingButton, 3, 0 );
    QObject::connect( startRecordingButton, &QPushButton::clicked,
        vf, &Viewfinder::startWriting );

    QPushButton* stopRecordingButton = new QPushButton( "Stop Recording" );
    layout->addWidget( stopRecordingButton, 3, 1 );
    QObject::connect( stopRecordingButton, &QPushButton::clicked,
        vf, &Viewfinder::stopWriting );

    QLabel* statusBar = new QLabel( "Idle" );
    layout->addWidget( statusBar, layout->rowCount(), 0, 2, 1 );
    QObject::connect( vf, &Viewfinder::statusChanged,
        statusBar, &QLabel::setText );

    QWidget mainWidget;
    mainWidget.setLayout( layout );
    mainWidget.move( 0, 0 );
    mainWidget.show();

    int exitCode = app.exec();

    delete vf;
    return exitCode;
}
