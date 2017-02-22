#include "libcgt/qt_interop/Filesystem.h"

#include <QDir>

namespace libcgt { namespace qt_interop {

bool mkpath( const std::string& path )
{
    QDir dir( QString::fromStdString( path ) );
    return dir.mkpath( "." );
}

} } // qt_interop, libcgt

