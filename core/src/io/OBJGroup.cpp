#include "io/OBJGroup.h"

#include <cassert>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
const std::vector< int > OBJGroup::s_invalidSentinel;

OBJGroup::OBJGroup( const std::string& name ) :

    m_name( name ),
    m_hasTextureCoordinates( false ),
    m_hasNormals( true )

{
    addMaterial( "" );
}

const std::string& OBJGroup::name() const
{
    return m_name;
}

bool OBJGroup::hasTextureCoordinates() const
{
    return m_hasTextureCoordinates;
}

void OBJGroup::setHasTextureCoordinates( bool b )
{
    m_hasTextureCoordinates = b;
}

bool OBJGroup::hasNormals() const
{
    return m_hasNormals;
}

void OBJGroup::setHasNormals( bool b )
{
    m_hasNormals = b;
}

int OBJGroup::numFaces() const
{
    return static_cast< int >( m_faces.size() );
}

const std::vector< OBJFace >& OBJGroup::faces() const
{
    return m_faces;
}

std::vector< OBJFace >& OBJGroup::faces()
{
    return m_faces;
}

void OBJGroup::addFace( const OBJFace& face )
{
    m_faces.push_back( face );

    const std::string& lastMaterial = m_materialNames.back();
    m_facesByMaterial[ lastMaterial ].push_back( numFaces() - 1 );
}

int OBJGroup::numMaterials() const
{
    return static_cast< int >( m_materialNames.size() );
}

const std::vector< std::string >& OBJGroup::materialNames() const
{
    return m_materialNames;
}

std::vector< std::string >& OBJGroup::materialNames()
{
    return m_materialNames;
}

void OBJGroup::addMaterial( const std::string& materialName )
{
    m_materialNames.push_back( materialName );

    if( m_facesByMaterial.find( materialName ) == m_facesByMaterial.end() )
    {
        m_facesByMaterial[ materialName ] = std::vector< int >();
    }
}

const std::vector< int >& OBJGroup::facesForMaterial(
    const std::string& materialName ) const
{
    auto itr = m_facesByMaterial.find( materialName );
    if( itr != m_facesByMaterial.end() )
    {
        return itr->second;
    }
    else
    {
        return s_invalidSentinel;
    }
}

std::vector< int >& OBJGroup::facesForMaterial(
    const std::string& materialName )
{
    return m_facesByMaterial[ materialName ];
}

const std::vector< int >& OBJGroup::facesForMaterial( int materialIndex ) const
{
    if( materialIndex < m_materialNames.size() )
    {
        return facesForMaterial( m_materialNames[ materialIndex ] );
    }
    else
    {
        return s_invalidSentinel;
    }
}

std::vector< int >& OBJGroup::facesForMaterial( int materialIndex )
{
    if( materialIndex < m_materialNames.size() )
    {
        return facesForMaterial( m_materialNames[ materialIndex ] );
    }
    else
    {
        return facesForMaterial( "" );
    }
}
