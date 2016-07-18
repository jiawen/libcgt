#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "GLSeparableProgram.h"

class GLProgramManager
{
public:

    GLProgramManager() = default;

    void addFromFile( const std::string& programName,
        GLSeparableProgram::Type type,
        const std::string& sourceFile );
    void addFromSourceCode( const std::string& programName,
        GLSeparableProgram::Type type,
        const std::string& sourceFile );

    GLSeparableProgram& get( const std::string& programName );

private:

    std::unordered_map< std::string, GLSeparableProgram > m_programsByName;

};
