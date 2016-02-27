#include <iomanip>
#include <sstream>

#include "io/NumberedFilenameBuilder.h"

NumberedFilenameBuilder::NumberedFilenameBuilder( const std::string& prefix,
                                                 const std::string& suffix,
                                                 int nDigits ) :
    m_prefix( prefix ),
    m_suffix( suffix ),
    m_nDigits( nDigits )
{
}

std::string NumberedFilenameBuilder::filenameForNumber( int number )
{
    std::stringstream numStream;
    numStream << std::setw( m_nDigits ) << std::setfill( '0' ) << number;
    return m_prefix + numStream.str() + m_suffix;
}
