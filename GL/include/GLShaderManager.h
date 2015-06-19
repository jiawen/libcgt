#pragma once

#include <memory>
#include <string>
#include <unordered_map>

class GLProgram;

class GLShaderManager
{
public:

    GLShaderManager() = default;

    // TODO: separate shader objects, program pipelines, etc
    bool add( const std::string& programName,
        const std::string& vertexShaderSourceFile,
        const std::string& fragmentShaderSourceFile );

    std::shared_ptr< GLProgram > get( const std::string& programName );

private:

    std::unordered_map< std::string, std::shared_ptr< GLProgram > > m_programsByName;

};