#include <QApplication>

#include "Viewfinder.h"

int main( int argc, char* argv[] )
{
    if( argc < 2 )
    {
        printf( "Usage: %s <--color|--infrared|--color-and-infrared>"
            " [dst_dir]\n", argv[0] );
        return 1;
    }
    QApplication app( argc, argv );

    int mode = -1;
    std::string modeString( argv[ 1 ] );
    if( modeString == "--color" )
    {
        mode = 0;
    }
    else if( modeString == "--infrared" )
    {
        mode = 1;
    }
    else if( modeString == "--color-and-infrared" )
    {
        mode = 2;
    }

    if( mode == -1 )
    {
        printf( "Usage: %s <--color|--infrared|--color-and-infrared>"
            " [dst_dir]\n", argv[ 0 ] );
        return 1;
    }

    // TODO:
    // [--start-after N]
    // [--stop-after 0] means unlimited

    std::string destDir = "";
    if( argc > 2 )
    {
        destDir = argv[ 2 ];
    }

    Viewfinder vf( mode, destDir );
    vf.move( 10, 10 );
    vf.show();

    return app.exec();
}
