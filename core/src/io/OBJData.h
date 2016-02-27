#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"

#include "io/OBJGroup.h"
#include "io/OBJMaterial.h"

class OBJData
{
public:

    OBJData() = default;

    // ----- raw flattened geometry -----
    const std::vector< Vector3f >& positions() const;
    std::vector< Vector3f >& positions();
    const std::vector< Vector2f >& textureCoordinates() const;
    std::vector< Vector2f >& textureCoordinates();
    const std::vector< Vector3f >& normals() const;
    std::vector< Vector3f >& normals();

    // ----- groups -----

    // Returns the number of groups.
    int numGroups() const;

    // Returns all the groups from OBJData (in file order).
    const std::vector< OBJGroup >& groups() const;
    std::vector< OBJGroup >& groups();

    // Adds a group and returns a reference to it.
    // If it already exists, returns the existing group.
    OBJGroup& addGroup( const std::string& groupName );

    bool containsGroup( const std::string& groupName ) const;

    // Returns a reference to the group if it exists.
    // Otherwise, returns a special invalid group with name "invalid".
    OBJGroup& getGroupByName( const std::string& groupName );

    // ----- materials -----

    // Returns the number of materials.
    int numMaterials() const;

    // returns all the materials from OBJData (in file order)
    std::vector< OBJMaterial >& materials();

    // Adds a new material by name and returns a reference to it.
    OBJMaterial& addMaterial( const std::string& name );

    bool containsMaterial( const std::string& name ) const;

    // Returns a reference to the material if it exists.
    // Otherwise, returns a special invalid material with name "invalid".
    OBJMaterial& getMaterialByName( const std::string& name );

    // ----- Data cleanup -----
    void removeEmptyGroups();

    // ----- I/O -----
    bool save( const std::string& filename );

private:

    std::vector< Vector3f > m_positions;
    std::vector< Vector2f > m_textureCoordinates;
    std::vector< Vector3f > m_normals;

    std::vector< OBJGroup > m_groups;
    std::unordered_map< std::string, int > m_groupIndicesByName;

    std::vector< OBJMaterial > m_materials;
    std::unordered_map< std::string, int > m_materialIndicesByName;

    // TODO: deal with const.
    static OBJGroup s_invalidGroup;
    static OBJMaterial s_invalidMaterial;
};
