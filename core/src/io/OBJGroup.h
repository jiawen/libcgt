#pragma once

#include <vector>
#include <QString>
#include <QHash>

#include "OBJFace.h"

class OBJGroup
{
public:

    OBJGroup( QString name );

    QString name() const;

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
    const std::vector< QString >& materialNames() const;
    std::vector< QString >& materialNames();

    // add a new material and sets it as current
    void addMaterial( QString materialName );

    std::vector< int >& facesForMaterial( QString materialName );

    std::vector< int >& facesForMaterial( int materialIndex );

private:

    QString m_name;
    bool m_hasTextureCoordinates;
    bool m_hasNormals;

    std::vector< QString > m_materialNames;
    QHash< QString, std::vector< int > > m_facesByMaterial;

    std::vector< OBJFace > m_faces;

};
