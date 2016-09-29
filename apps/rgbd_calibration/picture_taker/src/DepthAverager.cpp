#include <gflags/gflags.h>
#include <QApplication>
#include <QDir>

#include "DepthAveragerViewfinder.h"

DEFINE_string( output_dir, "",
    "Output directory. Writes depth_average_<#####>.<png>,"
    " depth_average_<#####>.<pfm> and infrared_average_<#####>.png"
    " to <output_dir>." );

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    QApplication app( argc, argv );

    if( FLAGS_output_dir == "" )
    {
        printf( "No destination dir specified: in DRY RUN mode.\n" );
    }
    else
    {
        printf( "Writing outputs to: %s\n", FLAGS_output_dir.c_str() );

        // Try to create destination directory if it doesn't exist.
        QDir dir( QString::fromStdString( FLAGS_output_dir ) );
        if( !dir.mkpath( "." ) )
        {
            fprintf( stderr, "Unable to create output directory: %s\n",
                FLAGS_output_dir.c_str() );
            return 1;
        }
    }

    printf( "Press D to save depth average.\n"
        "Press F to reset depth accumulator.\n"
        "Press I to save a infrared average.\n"
        "Press O to reset infrared accumulator.\n"
    );

    DepthAveragerViewfinder vf( FLAGS_output_dir );
    vf.move( 10, 10 );
    vf.show();

    return app.exec();
}
