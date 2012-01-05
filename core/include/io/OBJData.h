#pragma once

#include <QHash>
#include <QString>
#include <QVector>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>

#include "io/OBJGroup.h"
#include "io/OBJMaterial.h"

class OBJData
{
public:

	OBJData();
	virtual ~OBJData();

	QVector< Vector3f >* getPositions();
	QVector< Vector2f >* getTextureCoordinates();
	QVector< Vector3f >* getNormals();

	QHash< QString, OBJGroup* >* getGroups();

	// adds a group and returns a pointer to the group
	OBJGroup* addGroup( QString groupName );

	// returns a pointer to the group if it exists
	// returns nullptr otherwise
	OBJGroup* getGroup( QString groupName );

	bool containsGroup( QString groupName );

	// adds a group and returns a pointer to the group
	OBJMaterial* addMaterial( QString name );

	// returns a pointer to the group if it exists
	// returns nullptr otherwise
	OBJMaterial* getMaterial( QString name );

	bool containsMaterial( QString name );

private:

	QVector< Vector3f > m_positions;
	QVector< Vector2f > m_textureCoordinates;
	QVector< Vector3f > m_normals;

	QHash< QString, OBJGroup* > m_groups;
	QHash< QString, OBJMaterial* > m_materials;

};
