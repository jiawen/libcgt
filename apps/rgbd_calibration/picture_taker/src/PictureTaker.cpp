#include <gflags/gflags.h>
#include <QApplication>
#include <QDir>
#include <third_party/pystring/pystring.h>

#include "PictureTakerViewfinder.h"

static bool validateStartAfter( const char* flagname, gflags::int32 value )
{
    return( value > 0 );
}

static bool validateMode( const char* flagname, const std::string& mode )
{
    return
    (
        mode == "color" ||
        mode == "infrared" ||
        mode == "toggleColorInfrared"
    );
}

DEFINE_int32( start_after, 10, "Start saving after N seconds." );
static const bool start_after_dummy = gflags::RegisterFlagValidator(
    &FLAGS_start_after, &validateStartAfter );

DEFINE_string( mode, "color",
    "Operating mode: color|infrared|toggleColorInfrared" );
static const bool mode_dummy = gflags::RegisterFlagValidator(
    &FLAGS_mode, &validateMode );

DEFINE_int32( shot_interval, 5,
    "Wait N seconds between shots. "
    "If set to 0, takes a shot when spacebar is pressed." );
DEFINE_int32( secondary_shot_interval, 2,
    "In toggleColorInfrared mode, wait N seconds after the color shot before "
    "taking the infrared shot. Will be forced to 0 if shot_interval is 0." );

DEFINE_string( output_dir, "",
    "Output directory.\n"
    "In color or infrared mode, writes files to"
    " <output_dir>/<stream>/<stream>_<#####>.png.\n"
    "In toggleColorInfrared mode, writes color_<#####>.png and"
    " infrared_<#####>.png. to"
    " <output_dir>/infrared_color_stereo_images/.\n" );

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    if( FLAGS_shot_interval == 0 )
    {
        FLAGS_secondary_shot_interval = 0;
    }

    QApplication app( argc, argv );

    // TODO:
    // [--stop-after 0] means unlimited

    std::string streamOutputDir;

    if( FLAGS_output_dir == "" )
    {
        printf( "No destination dir specified: in DRY RUN mode.\n" );
        streamOutputDir = "";
    }
    else
    {
        printf( "Writing outputs to: %s\n", FLAGS_output_dir.c_str() );

        // Try to create destination directory if it doesn't exist.
        if( FLAGS_mode == "color" || FLAGS_mode == "infrared" )
        {
            streamOutputDir = pystring::os::path::join( FLAGS_output_dir,
                FLAGS_mode + "_images" );
        }
        else
        {
            streamOutputDir = pystring::os::path::join( FLAGS_output_dir,
                "infrared_color_stereo_images" );
        }

        QDir dir( QString::fromStdString( streamOutputDir ) );
        if( !dir.mkpath( "." ) )
        {
            fprintf( stderr, "Unable to create output directory: %s\n",
                streamOutputDir.c_str() );
            return 1;
        }
    }


    PictureTakerViewfinder vf( streamOutputDir );
    vf.move( 10, 10 );
    vf.show();

    return app.exec();
}
