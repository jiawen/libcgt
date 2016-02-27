#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "OBJFace.h"

class OBJGroup
{
public:

    OBJGroup( const std::string& name );

    const std::string& name() const;

    bool hasTextureCoordinates() const;
    void setHasTextureCoordinates( bool b );

    bool hasNormals() const;
    void setHasNormals( bool b );

    int numFaces() const;
    const std::vector< OBJFace >& faces() const;
    std::vector< OBJFace >& faces();

    // adds a new face to the current material
    void addFace( const OBJFace& face );

    int numMaterials() const;
    const std::vector< std::string >& materialNames() const;
    std::vector< std::string >& materialNames();

    // add a new material and sets it as current
    void addMaterial( const std::string& materialName );

    const std::vector< int >& facesForMaterial(
        const std::string& materialName ) const;
    std::vector< int >& facesForMaterial( const std::string& materialName );

    const std::vector< int >& facesForMaterial( int materialIndex ) const;
    std::vector< int >& facesForMaterial( int materialIndex );

private:

    std::string m_name;
    bool m_hasTextureCoordinates;
    bool m_hasNormals;

    std::vector< std::string > m_materialNames;
    std::unordered_map< std::string, std::vector< int > > m_facesByMaterial;

    std::vector< OBJFace > m_faces;

    static const std::vector< int > s_invalidSentinel;
};
