#pragma once

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
        ALL_STAGES = GL_ALL_SHADER_BITS,

        COMPUTE_SHADER_BIT = GL_COMPUTE_SHADER_BIT,
        FRAGMENT_SHADER_BIT = GL_FRAGMENT_SHADER_BIT,
#ifdef GL_PLATFORM_45
        GEOMETRY_SHADER_BIT = GL_GEOMETRY_SHADER_BIT,
        TESS_CONTROL_SHADER_BIT = GL_TESS_CONTROL_SHADER_BIT,
        TESS_EVALUATION_SHADER_BIT = GL_TESS_EVALUATION_SHADER_BIT,
#endif
        VERTEX_SHADER_BIT = GL_VERTEX_SHADER_BIT
    };

    GLProgramPipeline();
    GLProgramPipeline( GLProgramPipeline&& move );
    GLProgramPipeline& operator = ( GLProgramPipeline&& move );
    ~GLProgramPipeline();

    GLProgramPipeline( const GLProgramPipeline& copy ) = delete;
    GLProgramPipeline& operator = ( const GLProgramPipeline& copy ) = delete;

    GLuint id() const;

    // Attach the program to its corresponding stage.
    void attachProgram( const GLSeparableProgram& program );

    // Attach a program to the given stage.
    //
    // Note that although Stage is technically a bitfield and OpenGL lets you
    // attach multiple stages of a separable program to many pipeline stages at
    // once, it is not really necessary. GLSeparablePrograms are constructed
    // with a particular shader type and are not linked from traditional shader
    // objects. They contain at most one stage.
    //
    // Also note that using ALL_STAGES here is not what you expect, in that
    // it sets all stages to what is available in the program. If the program
    // does not have a vertex shader, it will set the pipeline's vertex shader
    // to null (rather than leaving it alone)!
    void attachProgram( const GLSeparableProgram& program, Stage stage );

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

    void destroy();
};
