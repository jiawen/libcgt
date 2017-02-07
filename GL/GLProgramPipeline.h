#pragma once

#include <memory>
#include <string>

#ifdef GL_PLATFORM_ES_31
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#endif
#ifdef GL_PLATFORM_45
#include <GL/glew.h>
#endif

class GLSeparableProgram;

class GLProgramPipeline
{
public:

    enum class Stage : GLbitfield
    {
        NO_STAGE = 0,

        COMPUTE_SHADER = GL_COMPUTE_SHADER_BIT,
        FRAGMENT_SHADER = GL_FRAGMENT_SHADER_BIT,
#ifdef GL_PLATFORM_45
        GEOMETRY_SHADER = GL_GEOMETRY_SHADER_BIT,
        TESS_CONTROL_SHADER = GL_TESS_CONTROL_SHADER_BIT,
        TESS_EVALUATION_SHADER = GL_TESS_EVALUATION_SHADER_BIT,
#endif
        VERTEX_SHADER = GL_VERTEX_SHADER
    };

    GLProgramPipeline();
    GLProgramPipeline( GLProgramPipeline&& move );
    GLProgramPipeline& operator = ( GLProgramPipeline&& move );
    ~GLProgramPipeline();

    GLProgramPipeline( const GLProgramPipeline& copy ) = delete;
    GLProgramPipeline& operator = ( const GLProgramPipeline& copy ) = delete;

    GLuint id() const;

    // Attach the program to its corresponding stage.
    void attachProgram( std::shared_ptr< GLSeparableProgram > program );

    // Retrieve the program attached to the given stage.
    std::shared_ptr< GLSeparableProgram > programAttachedAt( Stage stage );

    // Convenience function. Equivalent to
    // programAttachedAt( Stage::VERTEX_SHADER );
    std::shared_ptr< GLSeparableProgram > vertexProgram();

    // Convenience function. Equivalent to
    // programAttachedAt( Stage::FRAGMENT_SHADER );
    std::shared_ptr< GLSeparableProgram > fragmentProgram();

    // Detach any program that was attached to the given stage.
    void detachProgram( Stage stage );

    // Bind this pipeline to the rendering context;
    void bind();

    // Unbind all (the one and only) pipelines from the rendering context.
    static void unbindAll();

    // Returns true if the pipeline is valid.
    bool validate();

    // Retrieve the info log.
    std::string infoLog() const;

private:

    GLuint m_id;

    // TODO: this is somewhat brittle but I don't know of a better solution.
#ifdef GL_PLATFORM_45
    static constexpr int NUM_STAGES = 6;
#else
    static constexpr int NUM_STAGES = 3;
#endif
    std::shared_ptr< GLSeparableProgram > m_attachedPrograms[ NUM_STAGES ];

    void destroy();
};
