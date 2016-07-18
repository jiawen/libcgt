#include "GLProgramPipeline.h"

#include "GLSeparableProgram.h"

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

// virtual
GLProgramPipeline::~GLProgramPipeline()
{
    destroy();
}

GLuint GLProgramPipeline::id() const
{
    return m_id;
}

void GLProgramPipeline::attachProgram( const GLSeparableProgram& program )
{
    switch( program.type() )
    {
    case GLSeparableProgram::Type::COMPUTE_SHADER:
        attachProgram( program, Stage::COMPUTE_SHADER_BIT );
        break;
    case GLSeparableProgram::Type::FRAGMENT_SHADER:
        attachProgram( program, Stage::FRAGMENT_SHADER_BIT );
        break;
#ifdef GL_PLATFORM_45
    case GLSeparableProgram::Type::GEOMETRY_SHADER:
        attachProgram( program, Stage::GEOMETRY_SHADER_BIT );
        break;
    case GLSeparableProgram::Type::TESS_CONTROL_SHADER:
        attachProgram( program, Stage::TESS_CONTROL_SHADER_BIT );
        break;
    case GLSeparableProgram::Type::TESS_EVALUATION_SHADER:
        attachProgram( program, Stage::TESS_EVALUATION_SHADER_BIT );
        break;
#endif
    case GLSeparableProgram::Type::VERTEX_SHADER:
        attachProgram( program, Stage::VERTEX_SHADER_BIT );
        break;
    }
}

void GLProgramPipeline::attachProgram( const GLSeparableProgram& program,
    GLProgramPipeline::Stage stage )
{
    glUseProgramStages( id(), static_cast< GLbitfield >( stage ),
        program.id() );
}

void GLProgramPipeline::detachProgram( GLProgramPipeline::Stage stage )
{
    glUseProgramStages( id(), static_cast< GLbitfield >( stage ), 0 );
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
