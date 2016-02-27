#include "common/Parameters.h"

#include <fstream>
#include <pystring.h>

// static
Parameters* Parameters::instance()
{
    if( s_singleton == nullptr )
    {
        s_singleton = new Parameters();
    }

    return s_singleton;
}

// static
bool Parameters::parse( const std::string& filename )
{
    Parameters* p = Parameters::instance();

    // Attempt to read the file.
    std::ifstream inputFile( filename );
    if (inputFile.fail())
    {
        return false;
    }

    int lineNumber = 0;
    std::string line;
    const std::string delim( " " );

    while( std::getline( inputFile, line ) )
    {
        if( line == "" ||
            pystring::startswith( line, "#" ) ||
            pystring::startswith( line, "//" ) )
        {
            ++lineNumber;
            continue;
        }

        std::vector< std::string > tokens;
        pystring::split( line, tokens, delim );

        const std::string& type = tokens[0];
        if( type == "bool" )
        {
            const std::string& name = tokens[1];
            const std::string& value = tokens[2];
            bool v = ( value == "true" );
            p->setBool( name, v );
        }
        else if( type == "int" )
        {
            const std::string& name = tokens[1];
            const std::string& value = tokens[2];
            int v = std::stoi( value );
            p->setInt( name, v );
        }
        else if( type == "int[]" )
        {
            const std::string& name = tokens[1];
            std::vector< int > values;
            for( int i = 2; i < tokens.size(); ++i )
            {
                values.push_back( std::stoi( tokens[i] ) );
            }
            p->setIntArray( name, values );
        }
        else if( type == "float" )
        {
            const std::string& name = tokens[1];
            const std::string& value = tokens[2];
            float v = std::stof( value );
            p->setFloat( name, v );
        }
        else if( type == "float[]" )
        {
            const std::string& name = tokens[1];
            std::vector< float > values;
            for( int i = 2; i < tokens.size(); ++i )
            {
                values.push_back( std::stof( tokens[i] ) );
            }
            p->setFloatArray( name, values );
        }
        else if( type == "string" )
        {
            const std::string& name = tokens[1];
            const std::string& value = tokens[2];
            p->setString( name, value );
        }
        else if( type == "string[]" )
        {
            const std::string& name = tokens[1];
            std::vector< std::string > values;
            for( int i = 2; i < tokens.size(); ++i )
            {
                values.push_back( tokens[i] );
            }
            p->setStringArray( name, values );
        }
        else
        {
            fprintf( stderr, "Ignoring unknown type: %s\n", type.c_str() );
        }

        ++lineNumber;
    }

    bool succeeded = !( inputFile.fail() );
    inputFile.close();
    return succeeded;
}

bool Parameters::hasBool( const std::string& name )
{
    return m_boolParameters.find( name ) != m_boolParameters.end();
}

bool Parameters::getBool( const std::string& name )
{
    return m_boolParameters[ name ];
}

void Parameters::setBool( const std::string& name, bool value )
{
    m_boolParameters[ name ] = value;
}

void Parameters::toggleBool( const std::string& name )
{
    if( m_boolParameters.find( name ) != m_boolParameters.end() )
    {
        setBool( name, !getBool( name ) );
    }
}

bool Parameters::hasInt( const std::string& name )
{
    return m_intParameters.find( name ) != m_intParameters.end();
}

int Parameters::getInt( const std::string& name )
{
    return m_intParameters[ name ];
}

void Parameters::setInt( const std::string& name, int value )
{
    m_intParameters[ name ] = value;
}

bool Parameters::hasIntArray( const std::string& name )
{
    return m_intArrayParameters.find( name ) != m_intArrayParameters.end();
}

const std::vector< int >& Parameters::getIntArray( const std::string& name )
{
    return m_intArrayParameters[ name ];
}

void Parameters::setIntArray( const std::string& name,
                             const std::vector< int >& values )
{
    m_intArrayParameters[ name ] = values;
}

bool Parameters::hasFloat( const std::string& name )
{
    return m_floatParameters.find( name ) != m_floatParameters.end();
}

float Parameters::getFloat( const std::string& name )
{
    return m_floatParameters[ name ];
}

void Parameters::setFloat( const std::string& name, float value )
{
    m_floatParameters[ name ] = value;
}

bool Parameters::hasFloatArray( const std::string& name )
{
    return m_floatArrayParameters.find( name ) != m_floatArrayParameters.end();
}

const std::vector< float >& Parameters::getFloatArray( const std::string& name )
{
    return m_floatArrayParameters[ name ];
}

void Parameters::setFloatArray( const std::string& name,
                               const std::vector< float >& values )
{
    m_floatArrayParameters[ name ] = values;
}

bool Parameters::hasString( const std::string& name )
{
    return m_stringParameters.find( name ) != m_stringParameters.end();
}

const std::string& Parameters::getString( const std::string& name )
{
    return m_stringParameters[ name ];
}

void Parameters::setString( const std::string& name, const std::string& value )
{
    m_stringParameters[ name ] = value;
}

bool Parameters::hasStringArray( const std::string& name )
{
    return m_stringArrayParameters.find( name ) !=
        m_stringArrayParameters.end();
}

const std::vector< std::string >& Parameters::getStringArray(
    const std::string& name )
{
    return m_stringArrayParameters[ name ];
}

void Parameters::setStringArray( const std::string& name,
    const std::vector< std::string >& values )
{
    m_stringArrayParameters[ name ] = values;
}

Parameters::Parameters()
{

}

// static
Parameters* Parameters::s_singleton = nullptr;
