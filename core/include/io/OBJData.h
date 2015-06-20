#pragma once

#include <QHash>
#include <QString>

#include <vector>

#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"

#include "io/OBJGroup.h"
#include "io/OBJMaterial.h"

class OBJData
{
public:

    OBJData();
    virtual ~OBJData();

    // ----- raw flattened geometry -----
    std::vector< Vector3f >& positions();
    std::vector< Vector2f >& textureCoordinates();
    std::vector< Vector3f >& normals();

    // ----- groups -----

    // returns the number of groups
    int numGroups();

    // returns all the groups from OBJData (in file order)
    std::vector< OBJGroup >& groups();

    // adds a group and returns a reference to it
    // if it already exists, returns the existing group
    OBJGroup& addGroup( QString groupName );

    bool containsGroup( QString groupName );

    // returns a pointer to the group if it exists
    // returns nullptr otherwise
    OBJGroup* getGroupByName( QString groupName );

    // ----- materials -----

    // returns the number of materials
    int numMaterials();

    // returns all the materials from OBJData (in file order)
    std::vector< OBJMaterial >& materials();

    // adds a new material by name and returns a reference to it
    OBJMaterial& addMaterial( QString name );

    bool containsMaterial( QString name );

    // returns nullptr if it doesn't exist
    OBJMaterial* getMaterialByName( QString name );

    // ----- Data cleanup -----
    void removeEmptyGroups();

    // ----- I/O -----
    bool save( QString filename );

private:

    std::vector< Vector3f > m_positions;
    std::vector< Vector2f > m_textureCoordinates;
    std::vector< Vector3f > m_normals;

    std::vector< OBJGroup > m_groups;
    QHash< QString, OBJGroup* > m_groupsByName;

    std::vector< OBJMaterial > m_materials;
    QHash< QString, OBJMaterial* > m_materialsByName;

};
