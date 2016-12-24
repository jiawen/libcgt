#include <gflags/gflags.h>
#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>

#include "libcgt/core/vecmath/Box3f.h"
#include "libcgt/camera_wrappers/StreamConfig.h"

#include "Viewfinder.h"

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

// Can't have color and infrared at the same time.

static bool validateMode( const char* flagname, const std::string& mode )
{
    return
    (
        mode == "color" ||
        mode == "infrared" ||
        mode == "depth" ||
        mode == "color+depth" ||
        mode == "depth+infrared"
    );
}

DEFINE_string( mode, "color+depth",
    "Stream mode, must be one of: [color, depth, infrared, color+depth,"
    " depth+infrared]. Default: color+depth."
    " Note: color+infrared is not allowed." );
static const bool mode_dummy = gflags::RegisterFlagValidator(
    &FLAGS_mode, &validateMode );

DEFINE_string( output_dir, "",
    "Output dir. Files are written to <output_dir>/recording_<#####>.rgbd." );

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    if( FLAGS_output_dir == "" )
    {
        fprintf( stderr, "output_dir required.\n" );
        return 1;
    }
    // Try to create destination directory if it doesn't exist.
    QDir dir( QString::fromStdString( FLAGS_output_dir ) );
    if( !dir.mkpath( "." ) )
    {
        fprintf( stderr, "Unable to create output directory: %s\n",
            FLAGS_output_dir.c_str() );
        return 1;
    }

    QApplication app( argc, argv );

    std::vector< StreamConfig > config;
    if( FLAGS_mode == "color" )
    {
        config = COLOR_ONLY_CONFIG;
    }
    else if( FLAGS_mode == "infrared" )
    {
        config = INFRARED_ONLY_CONFIG;
    }
    else if( FLAGS_mode == "depth" )
    {
        config = DEPTH_ONLY_CONFIG;
    }
    else if( FLAGS_mode == "color+depth" )
    {
        config = COLOR_DEPTH_CONFIG;
    }
    else if( FLAGS_mode == "depth+infrared" )
    {
        config = DEPTH_INFRARED_CONFIG;
    }

    Viewfinder* vf = new Viewfinder( config, FLAGS_output_dir );

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
