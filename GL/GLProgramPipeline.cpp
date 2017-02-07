#include "GLProgramPipeline.h"

#include "GLSeparableProgram.h"

namespace
{

GLProgramPipeline::Stage shaderStageForType( GLSeparableProgram::Type type )
{
    switch( type )
    {
    case GLSeparableProgram::Type::COMPUTE_SHADER:
        return GLProgramPipeline::Stage::COMPUTE_SHADER;
    case GLSeparableProgram::Type::FRAGMENT_SHADER:
        return GLProgramPipeline::Stage::FRAGMENT_SHADER;
#ifdef GL_PLATFORM_45
    case GLSeparableProgram::Type::GEOMETRY_SHADER:
        return GLProgramPipeline::Stage::GEOMETRY_SHADER;
    case GLSeparableProgram::Type::TESS_CONTROL_SHADER:
        return GLProgramPipeline::Stage::TESS_CONTROL_SHADER;
    case GLSeparableProgram::Type::TESS_EVALUATION_SHADER:
        return GLProgramPipeline::Stage::TESS_EVALUATION_SHADER;
#endif
    case GLSeparableProgram::Type::VERTEX_SHADER:
        return GLProgramPipeline::Stage::VERTEX_SHADER;
    default:
        return GLProgramPipeline::Stage::NO_STAGE;
    }
}

GLbitfield glShaderStage( GLProgramPipeline::Stage stage )
{
    switch( stage )
    {
    case GLProgramPipeline::Stage::COMPUTE_SHADER:
        return GL_COMPUTE_SHADER_BIT;
    case GLProgramPipeline::Stage::FRAGMENT_SHADER:
        return GL_FRAGMENT_SHADER_BIT;
#ifdef GL_PLATFORM_45
    case GLProgramPipeline::Stage::GEOMETRY_SHADER:
        return GL_GEOMETRY_SHADER_BIT;
    case GLProgramPipeline::Stage::TESS_CONTROL_SHADER:
        return GL_TESS_CONTROL_SHADER_BIT;
    case GLProgramPipeline::Stage::TESS_EVALUATION_SHADER:
        return GL_TESS_EVALUATION_SHADER_BIT;
#endif
    case GLProgramPipeline::Stage::VERTEX_SHADER:
        return GL_VERTEX_SHADER_BIT;
    default:
        return 0;
    }
}

int attachedProgramArrayIndexForStage( GLProgramPipeline::Stage stage )
{
    switch( stage )
    {
#ifdef GL_PLATFORM_45
    case GLProgramPipeline::Stage::COMPUTE_SHADER:
        return 0;
    case GLProgramPipeline::Stage::FRAGMENT_SHADER:
        return 1;
    case GLProgramPipeline::Stage::GEOMETRY_SHADER:
        return 2;
    case GLProgramPipeline::Stage::TESS_CONTROL_SHADER:
        return 3;
    case GLProgramPipeline::Stage::TESS_EVALUATION_SHADER:
        return 4;
    case GLProgramPipeline::Stage::VERTEX_SHADER:
        return 5;
#else
    case GLProgramPipeline::Stage::COMPUTE_SHADER:
        return 0;
    case GLProgramPipeline::Stage::FRAGMENT_SHADER:
        return 1;
    case GLProgramPipeline::Stage::VERTEX_SHADER:
        return 2;
#endif
    default:
        return -1;
    }
}

}

GLProgramPipeline::GLProgramPipeline()
{
    // These behave exactly the same way, but we might as well use the DSA form
    // while it's available.
#ifdef GL_PLATFORM_45
    glCreateProgramPipelines( 1, &m_id );
#endif

#ifdef GL_PLATFORM_ES_31
    glGenProgramPipelines( 1, &m_id );
#endif
}

GLProgramPipeline::GLProgramPipeline( GLProgramPipeline&& move )
{
    destroy();
    m_id = move.m_id;
    move.m_id = 0;
}

GLProgramPipeline& GLProgramPipeline::operator = (
    GLProgramPipeline&& move )
{
    if( this != &move )
    {
        destroy();
        m_id = move.m_id;
        move.m_id = 0;
    }
    return *this;
}

GLProgramPipeline::~GLProgramPipeline()
{
    destroy();
}

GLuint GLProgramPipeline::id() const
{
    return m_id;
}

void GLProgramPipeline::attachProgram(
    std::shared_ptr< GLSeparableProgram > program )
{
    if( program->type() == GLSeparableProgram::Type::NO_TYPE )
    {
        return;
    }

    GLProgramPipeline::Stage stage = shaderStageForType( program->type() );
    GLbitfield glStage = glShaderStage( stage );
    glUseProgramStages( id(), glStage, program->id() );

    // Add the shared pointer to the array of attached programs.
    int idx = attachedProgramArrayIndexForStage( stage );
    m_attachedPrograms[ idx ] = program;
}

std::shared_ptr< GLSeparableProgram > GLProgramPipeline::programAttachedAt(
    Stage stage )
{
    if( stage == GLProgramPipeline::Stage::NO_STAGE )
    {
        return nullptr;
    }

    int idx = attachedProgramArrayIndexForStage( stage );
    return m_attachedPrograms[ idx ];
}

std::shared_ptr< GLSeparableProgram > GLProgramPipeline::vertexProgram()
{
    return programAttachedAt( GLProgramPipeline::Stage::VERTEX_SHADER );
}

std::shared_ptr< GLSeparableProgram > GLProgramPipeline::fragmentProgram()
{
    return programAttachedAt(
        GLProgramPipeline::Stage::FRAGMENT_SHADER );
}

void GLProgramPipeline::detachProgram( GLProgramPipeline::Stage stage )
{
    if( stage == GLProgramPipeline::Stage::NO_STAGE )
    {
        return;
    }

    // Tell GL to detach the program.
    GLbitfield glStage = glShaderStage( stage );
    glUseProgramStages( id(), glStage, 0 );

    // Remove the shared pointer from the array of attached programs.
    int idx = attachedProgramArrayIndexForStage( stage );
    m_attachedPrograms[ idx ] = nullptr;
}

void GLProgramPipeline::bind()
{
    glBindProgramPipeline( id() );
}

// static
void GLProgramPipeline::unbindAll()
{
    glBindProgramPipeline( 0 );
}

bool GLProgramPipeline::validate()
{
    // Call glValidateProgramPipeline(), which puts a bool into the program
    // status.
    glValidateProgramPipeline( id() );
    // Now retrieve the bit and return it.
    GLint val;
    glGetProgramPipelineiv( id(), GL_VALIDATE_STATUS, &val );
    return( val == GL_TRUE );
}

std::string GLProgramPipeline::infoLog() const
{
    GLint length;
    glGetProgramPipelineiv( id(), GL_INFO_LOG_LENGTH, &length );
    if( length > 1 )
    {
        std::string log( length - 1, '\0' );
        char* bufferStart = &( log[ 0 ] );
        GLsizei tmp;
        glGetProgramPipelineInfoLog( id(), length, &tmp, bufferStart );
        return log;
    }
    else
    {
        return std::string();
    }
}

void GLProgramPipeline::destroy()
{
    if( m_id != 0 )
    {
        glDeleteProgramPipelines( 1, &m_id );
        m_id = 0;
    }
}
