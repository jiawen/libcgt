#include "GLShaderManager.h"

#include "GLProgram.h"
#include "GLShader.h"
   
bool GLShaderManager::add( const std::string& programName,
    const std::string& vertexShaderSourceFile,
    const std::string& fragmentShaderSourceFile )
{
    GLShader* pVS;
    GLShader* pFS;

    pVS = GLShader::vertexShaderFromSourceFile( vertexShaderSourceFile.c_str() );
    pFS = GLShader::fragmentShaderFromSourceFile( fragmentShaderSourceFile.c_str() );

    GLProgram* pProgram = GLProgram::fromShaders( { pVS, pFS } );

    delete pFS;
    delete pVS;

    if( pProgram != nullptr )
    {
        std::shared_ptr< GLProgram > p( pProgram );
        m_programsByName[ programName ] = p;
        return true;
    }
    return false;
}

std::shared_ptr< GLProgram > GLShaderManager::get( const std::string& programName )
{
    return m_programsByName[ programName ];
}