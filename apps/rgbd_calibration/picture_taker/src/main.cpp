#include <QApplication>

#include "Viewfinder.h"

int main( int argc, char* argv[] )
{
    if( argc < 2 )
    {
        printf( "Usage: %s <directory>\n", argv[0] );
        return 1;
    }
    QApplication app( argc, argv );

    Viewfinder vf( argv[1] );
    vf.move( 10, 10 );
    vf.show();

    return app.exec();
}
