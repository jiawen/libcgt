#include "io/OBJData.h"

// static
OBJGroup OBJData::s_invalidGroup( "invalid" );

// static
OBJMaterial OBJData::s_invalidMaterial( "invalid" );

const std::vector< Vector3f >& OBJData::positions() const
{
    return m_positions;
}

std::vector< Vector3f >& OBJData::positions()
{
    return m_positions;
}

const std::vector< Vector2f >& OBJData::textureCoordinates() const
{
    return m_textureCoordinates;
}

std::vector< Vector2f >& OBJData::textureCoordinates()
{
    return m_textureCoordinates;
}

const std::vector< Vector3f >& OBJData::normals() const
{
    return m_normals;
}

std::vector< Vector3f >& OBJData::normals()
{
    return m_normals;
}

int OBJData::numGroups() const
{
    return static_cast< int >( m_groups.size() );
}

const std::vector< OBJGroup >& OBJData::groups() const
{
    return m_groups;
}

std::vector< OBJGroup >& OBJData::groups()
{
    return m_groups;
}

OBJGroup& OBJData::addGroup( const std::string& name )
{
    if( m_groupIndicesByName.find( name ) == m_groupIndicesByName.end() )
    {
        m_groupIndicesByName[ name ] = static_cast< int >( m_groups.size() );
        m_groups.emplace_back( name );
    }

    return m_groups.back();
}

bool OBJData::containsGroup( const std::string& name ) const
{
    return m_groupIndicesByName.find( name ) != m_groupIndicesByName.end();
}

OBJGroup& OBJData::getGroupByName( const std::string& name )
{
    if( containsGroup( name ) )
    {
        return m_groups[ m_groupIndicesByName[ name ] ];
    }
    else
    {
        return s_invalidGroup;
    }
}

int OBJData::numMaterials() const
{
    return static_cast< int >( m_materials.size() );
}

std::vector< OBJMaterial >& OBJData::materials()
{
    return m_materials;
}

OBJMaterial& OBJData::addMaterial( const std::string& name )
{
    if( !containsMaterial( name ) )
    {
        m_materialIndicesByName[ name ] =
            static_cast< int >( m_materials.size() );
        m_materials.emplace_back( name );
    }

    return m_materials.back();
}

bool OBJData::containsMaterial( const std::string& name ) const
{
    return m_materialIndicesByName.find( name ) !=
        m_materialIndicesByName.end();
}

OBJMaterial& OBJData::getMaterialByName( const std::string& name )
{
    if( containsMaterial( name ) )
    {
        return m_materials[ m_materialIndicesByName[ name ] ];
    }
    else
    {
        return s_invalidMaterial;
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

    // Count how many were shifted over.
    int nRemoved = 0;
    for( int i = 0; i < static_cast< int >( removeList.size() ); ++i )
    {
        int groupIndex = removeList[i];
        int offsetGroupIndex = groupIndex - nRemoved;

        const std::string& groupName = m_groups[i].name();
        m_groups.erase( m_groups.begin() + offsetGroupIndex );
    }

    for( int i = 0; i < static_cast< int >( m_groups.size()); ++i )
    {
        m_groupIndicesByName[ m_groups[ i ].name() ] = i;
    }
}

bool OBJData::save( const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == nullptr )
    {
        return false;
    }

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
