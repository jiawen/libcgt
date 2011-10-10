#ifndef OBJ_GROUP_H
#define OBJ_GROUP_H

#include <QString>
#include <QVector>

#include "OBJFace.h"

// TODO: getFaces(), etc: don't return pointers

class OBJGroup
{
public:

	OBJGroup( QString name );

	QString getName();
	QVector< OBJFace >* getFaces();

	bool hasTextureCoordinates();
	void setHasTextureCoordinates( bool b );

	bool hasNormals();
	void setHasNormals( bool b );

private:

	QString m_qsName;
	bool m_bHasTextureCoordinates;
	bool m_bHasNormals;
	QVector< OBJFace > m_qvFaces;

};

#endif // OBJ_GROUP_H
