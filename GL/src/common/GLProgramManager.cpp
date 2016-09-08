#include "GLProgramManager.h"

void GLProgramManager::addFromFile( const std::string& programName,
    GLSeparableProgram::Type type, const std::string& sourceFile )
{
    m_programsByName.emplace( programName,
        GLSeparableProgram::fromFile( type, sourceFile.c_str() ) );
}

void GLProgramManager::addFromSourceCode( const std::string& programName,
    GLSeparableProgram::Type type,
    const std::string& sourceCode )
{
    m_programsByName.emplace( programName,
        GLSeparableProgram( type, sourceCode.c_str() ) );
}

GLSeparableProgram& GLProgramManager::get( const std::string& programName )
{
    return m_programsByName[ programName ];
}
