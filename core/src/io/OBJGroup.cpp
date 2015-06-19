#include "io/OBJGroup.h"

#include <cassert>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

OBJGroup::OBJGroup( QString name ) :

	m_name( name ),
    m_hasTextureCoordinates( false ),
	m_hasNormals( true )

{
	addMaterial( "" );
}

QString OBJGroup::name() const
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

	QString lastMaterial = m_materialNames.back();
	m_facesByMaterial[ lastMaterial ].push_back( numFaces() - 1 );
}

int OBJGroup::numMaterials() const
{
	return static_cast< int >( m_materialNames.size() );
}

const std::vector< QString >& OBJGroup::materialNames() const
{
	return m_materialNames;
}

std::vector< QString >& OBJGroup::materialNames()
{
	return m_materialNames;
}

void OBJGroup::addMaterial( QString materialName )
{
	m_materialNames.push_back( materialName );

	if( !( m_facesByMaterial.contains( materialName ) ) )
	{
		m_facesByMaterial.insert( materialName, std::vector< int >() );
	}
}

std::vector< int >& OBJGroup::facesForMaterial( QString materialName )
{
	assert( m_facesByMaterial.contains( materialName ) );
	std::vector< int >& output = m_facesByMaterial[ materialName ];
	return output;
}

std::vector< int >& OBJGroup::facesForMaterial( int materialIndex )
{
	assert( materialIndex < m_materialNames.size() );
	return facesForMaterial( m_materialNames[ materialIndex ] );
}
