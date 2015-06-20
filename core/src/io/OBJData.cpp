#include "io/OBJData.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

OBJData::OBJData()
{

}

// virtual
OBJData::~OBJData()
{

}

std::vector< Vector3f >& OBJData::positions()
{
    return m_positions;
}

std::vector< Vector2f >& OBJData::textureCoordinates()
{
    return m_textureCoordinates;
}

std::vector< Vector3f >& OBJData::normals()
{
    return m_normals;
}

int OBJData::numGroups()
{
    return static_cast< int >( m_groups.size() );
}

std::vector< OBJGroup >& OBJData::groups()
{
    return m_groups;
}

OBJGroup& OBJData::addGroup( QString name )
{
    if( !( m_groupsByName.contains( name ) ) )
    {
        m_groups.push_back( OBJGroup( name ) );
        m_groupsByName.insert( name, &( m_groups.back() ) );
    }

    return m_groups.back();
}

bool OBJData::containsGroup( QString name )
{
    return m_groupsByName.contains( name );
}

OBJGroup* OBJData::getGroupByName( QString name )
{
    if( containsGroup( name ) )
    {
        return m_groupsByName[ name ];
    }
    else
    {
        return nullptr;
    }
}

int OBJData::numMaterials()
{
    return static_cast< int >( m_materials.size() );
}

std::vector< OBJMaterial >& OBJData::materials()
{
    return m_materials;
}

OBJMaterial& OBJData::addMaterial( QString name )
{
    if( !containsMaterial( name ) )
    {
        m_materials.push_back( OBJMaterial( name ) );
        m_materialsByName.insert( name, &( m_materials.back() ) );
    }

    return m_materials.back();
}

bool OBJData::containsMaterial( QString name )
{
    return m_materialsByName.contains( name );
}

OBJMaterial* OBJData::getMaterialByName( QString name )
{
    if( containsMaterial( name ) )
    {
        return m_materialsByName[ name ];
    }
    else
    {
        return nullptr;
    }
}

void OBJData::removeEmptyGroups()
{
    std::vector< int > removeList;

    for( int i = 0; i < numGroups(); ++i )
    {
        if( m_groups[i].numFaces() == 0 )
        {
            removeList.push_back( i );
        }
    }

    // count how many were shifted over
    int nRemoved = 0;
    for( int i = 0; i < static_cast< int >( removeList.size() ); ++i )
    {
        int groupIndex = removeList[i];
        int offsetGroupIndex = groupIndex - nRemoved;

        QString groupName = m_groups[i].name();
        m_groups.erase( m_groups.begin() + offsetGroupIndex );
        m_groupsByName.remove( groupName );
    }

    // stuff moved in memory, re-index
    if( removeList.size() > 0 )
    {
        for( int i = 0; i < numGroups(); ++i )
        {
            QString groupName = m_groups[i].name();
            m_groupsByName[ groupName ] = &( m_groups[i] );
        }
    }
}

bool OBJData::save( QString filename )
{
    QByteArray ba = filename.toUtf8();
    FILE* fp = fopen( ba.data(), "w" );

    for( size_t i = 0; i < m_positions.size(); ++i )
    {
        Vector3f v = m_positions[ i ];
        fprintf( fp, "v %f %f %f\n", v.x, v.y, v.z );
    }

    for (size_t i = 0; i < m_textureCoordinates.size(); ++i)
    {
        Vector2f t = m_textureCoordinates[ i ];
        fprintf( fp, "vt %f %f\n", t.x, t.y );
    }

    for (size_t i = 0; i < m_normals.size(); ++i)
    {
        Vector3f n = m_normals[ i ];
        fprintf( fp, "vn %f %f %f\n", n.x, n.y, n.z );
    }

    for( int g = 0; g < numGroups(); ++g )
    {
        OBJGroup& group = m_groups[ g ];
        const auto& groupFaces = group.faces();

        for( int m = 0; m < group.numMaterials(); ++m )
        {
            const auto& faceIndices = group.facesForMaterial( m );
            for( int i = 0; i < static_cast< int >( faceIndices.size() ); ++i )
            {
                const OBJFace& face = groupFaces[ faceIndices[ i ] ];

                const auto& pis = face.positionIndices();
                const auto& tis = face.textureCoordinateIndices();
                const auto& nis = face.normalIndices();

                int nIndices = static_cast< int >( pis.size() );
                fprintf( fp, "f" );
                for( int j = 0; j < nIndices; ++j )
                {
                    int pi = pis[ j ] + 1;
                    fprintf( fp, " %d", pi );

                    if( group.hasTextureCoordinates() )
                    {
                        int ti = tis[ j ] + 1;
                        if( group.hasNormals() )
                        {
                            int ni = nis[ j ] + 1;
                            fprintf( fp, "/%d/%d", ti, ni );
                        }
                        else
                        {
                            fprintf( fp, "/%d", ti );
                        }
                    }
                    else if( group.hasNormals() )
                    {
                        int ni = nis[ j ] + 1;
                        fprintf( fp, "//%d", ni );
                    }
                }
                fprintf( fp, "\n" );
            }
        }
    }

    fclose( fp );

    // TODO: handle errors in writing
    return true;
}
