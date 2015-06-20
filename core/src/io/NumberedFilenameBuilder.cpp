#include "io/NumberedFilenameBuilder.h"

NumberedFilenameBuilder::NumberedFilenameBuilder( QString prefix, QString suffix, int nDigits ) :

    m_nDigits( nDigits )

{
    m_baseString = prefix + "%1" + suffix;
}

QString NumberedFilenameBuilder::filenameForNumber( int number )
{
    return m_baseString.arg( number, m_nDigits, 10, QChar( '0' ) );
}
