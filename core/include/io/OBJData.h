#ifndef OBJ_DATA_H
#define OBJ_DATA_H

#include <QHash>
#include <QString>
#include <QVector>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>

#include "io/OBJGroup.h"

class OBJData
{
public:

	OBJData();
	// TODO: destructor!

	QVector< Vector3f >* getPositions();
	QVector< Vector2f >* getTextureCoordinates();
	QVector< Vector3f >* getNormals();

	QHash< QString, OBJGroup* >* getGroups();

	// adds a group and returns a pointer to the group
	OBJGroup* addGroup( QString groupName );

	// returns a pointer to the group if it exists
	// returns NULL otherwise
	OBJGroup* getGroup( QString groupName );

	bool containsGroup( QString groupName );

private:

	QVector< Vector3f > m_qvPositions;
	QVector< Vector2f > m_qvTextureCoordinates;
	QVector< Vector3f > m_qvNormals;

	QHash< QString, OBJGroup* > m_qhGroups;

};

#endif // OBJ_DATA_H
