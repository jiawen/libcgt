#include "io/OBJMaterial.h"

OBJMaterial::OBJMaterial( const std::string& name ) :

    m_name( name ),
    m_illuminationModel( OBJMaterial::IlluminationModel::NONE ),

    m_ka( 0.2f, 0.2f, 0.2f ),
    m_kd( 0.8f, 0.8f, 0.8f ),
    m_ks( 1.0f, 1.0f, 1.0f ),

    m_d( 1 ),
    m_ns( 0 )

{

}

const std::string& OBJMaterial::name() const
{
    return m_name;
}

void OBJMaterial::setName( const std::string& name )
{
    m_name = name;
}

Vector3f OBJMaterial::ambientColor() const
{
    return m_ka;
}

void OBJMaterial::setAmbientColor( const Vector3f& color )
{
    m_ka = color;
}

Vector3f OBJMaterial::diffuseColor() const
{
    return m_kd;
}

void OBJMaterial::setDiffuseColor( const Vector3f& color )
{
    m_kd = color;
}

Vector3f OBJMaterial::specularColor() const
{
    return m_ks;
}

void OBJMaterial::setSpecularColor( const Vector3f& color )
{
    m_ks = color;
}

float OBJMaterial::alpha() const
{
    return m_d;
}

void OBJMaterial::setAlpha( float a )
{
    m_d = a;
}

float OBJMaterial::shininess() const
{
    return m_ns;
}

void OBJMaterial::setShininess( float s )
{
    m_ns = s;
}

const std::string& OBJMaterial::ambientTexture() const
{
    return m_mapKa;
}

void OBJMaterial::setAmbientTexture( const std::string& filename )
{
    m_mapKa = filename;
}

const std::string& OBJMaterial::diffuseTexture() const
{
    return m_mapKd;
}

void OBJMaterial::setDiffuseTexture( const std::string& filename )
{
    m_mapKd = filename;
}

OBJMaterial::IlluminationModel OBJMaterial::illuminationModel() const
{
    return m_illuminationModel;
}

void OBJMaterial::setIlluminationModel( OBJMaterial::IlluminationModel im )
{
    m_illuminationModel = im;
}