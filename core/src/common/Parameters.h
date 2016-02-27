#pragma once

// TODO: non-singleton
// TODO: move to IO

#include <string>
#include <vector>
#include <unordered_map>

class Parameters
{
public:

    // returns the singleton
    static Parameters* instance();

    // parses parameters in filename and overwrites it in the singleton
    static bool parse( const std::string& filename );

    bool hasBool( const std::string& name );
    bool getBool( const std::string& name );
    void setBool( const std::string& name, bool value );
    void toggleBool( const std::string& name );

    bool hasInt( const std::string& name );
    int getInt( const std::string& name );
    void setInt( const std::string& name, int value );

    bool hasIntArray( const std::string& name );
    const std::vector< int >& getIntArray( const std::string& name );
    void setIntArray( const std::string& name,
                     const std::vector< int >& values );

    bool hasFloat( const std::string& name );
    float getFloat( const std::string& name );
    void setFloat( const std::string& name, float value );

    bool hasFloatArray( const std::string& name );
    const std::vector< float >& getFloatArray( const std::string& name );
    void setFloatArray( const std::string& name,
                       const std::vector< float >& values );

    bool hasString( const std::string& name );
    const std::string& getString( const std::string& name );
    void setString( const std::string& name, const std::string& value );

    bool hasStringArray( const std::string& name );
    const std::vector< std::string >& getStringArray(
        const std::string& name );
    void setStringArray( const std::string& name,
        const std::vector< std::string >& values );

    // TODO: implement toString(). Or ostream& operator <<.

private:

    Parameters();

    static Parameters* s_singleton;

    std::unordered_map< std::string, bool > m_boolParameters;

    std::unordered_map< std::string, int > m_intParameters;
    std::unordered_map< std::string, std::vector< int > > m_intArrayParameters;

    std::unordered_map< std::string, float > m_floatParameters;
    std::unordered_map<
        std::string, std::vector< float > > m_floatArrayParameters;

    std::unordered_map< std::string, std::string > m_stringParameters;
    std::unordered_map<
        std::string, std::vector< std::string > > m_stringArrayParameters;
};
