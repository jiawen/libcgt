#pragma once

#include <string>
#include <vector>

#include "vecmath/Vector3f.h"

class OBJMaterial
{
public:

    enum class IlluminationModel
    {
        NONE = 0,
        DIFFUSE = 1,
        DIFFUSE_AND_SPECULAR = 2
    };

    OBJMaterial( const std::string& name = "" );

    const std::string& name() const;
    void setName( const std::string& name );

    Vector3f ambientColor() const;
    void setAmbientColor( const Vector3f& color );

    Vector3f diffuseColor() const;
    void setDiffuseColor( const Vector3f& color );

    Vector3f specularColor() const;
    void setSpecularColor( const Vector3f& color );

    float alpha() const;
    void setAlpha( float a );

    float shininess() const;
    void setShininess( float s );

    const std::string& ambientTexture() const;
    void setAmbientTexture( const std::string& filename );

    const std::string& diffuseTexture() const;
    void setDiffuseTexture( const std::string& filename );

    IlluminationModel illuminationModel() const;
    void setIlluminationModel( IlluminationModel im );

private:

    // required
    std::string m_name;
    IlluminationModel m_illuminationModel;

    Vector3f m_ka;
    Vector3f m_kd;
    Vector3f m_ks;

    float m_d; // alpha
    // float m_tr; // 1 - alpha
    float m_ns; // shininess

    std::string m_mapKa; // ambient texture
    std::string m_mapKd; // diffuse texture

    // TODO: parse others
    // http://www.fileformat.info/format/material/
};
