#include <gflags/gflags.h>
#include <QApplication>

#include "DepthAveragerViewfinder.h"

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    QApplication app( argc, argv );

    std::string destDir = "";
    if( argc > 1 )
    {
        destDir = argv[ 1 ];
        printf( "Writing outputs to: %s\n", destDir.c_str() );
        printf( "Press D to save depth average.\n"
            "Press F to reset depth accumulator.\n"
            "Press I to save a infrared average.\n"
            "Press O to reset infrared accumulator.\n"
        );
    }
    else
    {
        printf( "No destination dir specified: in DRY RUN mode.\n" );
    }

    DepthAveragerViewfinder vf( destDir );
    vf.move( 10, 10 );
    vf.show();

    return app.exec();
}
