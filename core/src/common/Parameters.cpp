#include "common/Parameters.h"

#include <QFile>
#include <QDataStream>
#include <QTextStream>
#include <QStringList>

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
bool Parameters::parse( QString filename )
{
    Parameters* p = Parameters::instance();

    // attempt to read the file
    QFile inputFile( filename );
    if( !( inputFile.open( QIODevice::ReadOnly ) ) )
    {
        return false;
    }

    int lineNumber = 0;
    QString line = "";
    QString delim( " " );

    QTextStream inputTextStream( &inputFile );
    line = inputTextStream.readLine();
    while( !( line.isNull() ) )
    {
        if( line == "" ||
            line.startsWith( "#" ) ||
            line.startsWith( "//" ) )
        {
            ++lineNumber;
            line = inputTextStream.readLine();
            continue;
        }

        QStringList tokens = line.split( delim, QString::SkipEmptyParts );

        QString type = tokens[0];

        if( type == "bool" )
        {
            QString name = tokens[1];
            QString value = tokens[2];
            bool v = ( value == "true" );
            p->setBool( name, v );
        }
        else if( type == "int" )
        {
            QString name = tokens[1];
            QString value = tokens[2];
            int v = value.toInt();
            p->setInt( name, v );
        }
        else if( type == "int[]" )
        {
            QString name = tokens[1];
            QVector< int > values;
            for( int i = 2; i < tokens.size(); ++i )
            {
                values.append( tokens[i].toInt() );
            }
            p->setIntArray( name, values );
        }
        else if( type == "float" )
        {
            QString name = tokens[1];
            QString value = tokens[2];
            float v = value.toFloat();
            p->setFloat( name, v );
        }
        else if( type == "float[]" )
        {
            QString name = tokens[1];
            QVector< float > values;
            for( int i = 2; i < tokens.size(); ++i )
            {
                values.append( tokens[i].toFloat() );
            }
            p->setFloatArray( name, values );
        }
        else if( type == "string" )
        {
            QString name = tokens[1];
            QString value = tokens[2];
            p->setString( name, value );
        }
        else if( type == "string[]" )
        {
            QString name = tokens[1];
            QVector< QString > values;
            for( int i = 2; i < tokens.size(); ++i )
            {
                values.append( tokens[i] );
            }
            p->setStringArray( name, values );
        }
        else
        {
            printf( "Ignoring unknown type: %s\n", qPrintable( type ) );
        }

        ++lineNumber;
        line = inputTextStream.readLine();
    }
    return true;
}

bool Parameters::hasBool( QString name )
{
    return m_boolParameters.contains( name );
}

bool Parameters::getBool( QString name )
{
    return m_boolParameters[ name ];
}

void Parameters::setBool( QString name, bool value )
{
    m_boolParameters[ name ] = value;
}

void Parameters::toggleBool( QString name )
{
    if( m_boolParameters.contains( name ) )
    {
        setBool( name, !getBool( name ) );
    }
}

bool Parameters::hasInt( QString name )
{
    return m_intParameters.contains( name );
}

int Parameters::getInt( QString name )
{
    return m_intParameters[ name ];
}

void Parameters::setInt( QString name, int value )
{
    m_intParameters[ name ] = value;
}

bool Parameters::hasIntArray( QString name )
{
    return m_intArrayParameters.contains( name );
}

QVector< int > Parameters::getIntArray( QString name )
{
    return m_intArrayParameters[ name ];
}

void Parameters::setIntArray( QString name, QVector< int > values )
{
    m_intArrayParameters[ name ] = values;
}

bool Parameters::hasFloat( QString name )
{
    return m_floatParameters.contains( name );
}

float Parameters::getFloat( QString name )
{
    return m_floatParameters[ name ];
}

void Parameters::setFloat( QString name, float value )
{
    m_floatParameters[ name ] = value;
}

bool Parameters::hasFloatArray( QString name )
{
    return m_floatArrayParameters.contains( name );
}

QVector< float > Parameters::getFloatArray( QString name )
{
    return m_floatArrayParameters[ name ];
}

void Parameters::setFloatArray( QString name, QVector< float > values )
{
    m_floatArrayParameters[ name ] = values;
}

bool Parameters::hasString( QString name )
{
    return m_stringParameters.contains( name );
}

QString Parameters::getString( QString name )
{
    return m_stringParameters[ name ];
}

void Parameters::setString( QString name, QString value )
{
    m_stringParameters[ name ] = value;
}

bool Parameters::hasStringArray( QString name )
{
    return m_stringArrayParameters.contains( name );
}

QVector< QString > Parameters::getStringArray( QString name )
{
    return m_stringArrayParameters[ name ];
}

void Parameters::setStringArray( QString name, QVector< QString > values )
{
    m_stringArrayParameters[ name ] = values;
}

Parameters::Parameters()
{

}

QDataStream& operator << ( QDataStream& s, const Parameters& p )
{
    s << p.m_boolParameters;
    s << p.m_intParameters;
    s << p.m_intArrayParameters;
    s << p.m_floatParameters;
    s << p.m_floatArrayParameters;
    s << p.m_stringParameters;
    s << p.m_stringArrayParameters;
    return s;
}

QDataStream& operator >> ( QDataStream& s, Parameters& p )
{
    s >> p.m_boolParameters;
    s >> p.m_intParameters;
    s >> p.m_intArrayParameters;
    s >> p.m_floatParameters;
    s >> p.m_floatArrayParameters;
    s >> p.m_stringParameters;
    s >> p.m_stringArrayParameters;
    return s;
}

// static
Parameters* Parameters::s_singleton = nullptr;
