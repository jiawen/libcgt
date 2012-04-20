#ifndef OBJ_FACE_H
#define OBJ_FACE_H

#include <QVector>

class OBJFace
{
public:

	OBJFace();
	OBJFace( bool hasTextureCoordinates, bool hasNormals );

	bool hasTextureCoordinates() const;
	bool hasNormals() const;

	int numVertices() const;

	QVector< int >* getPositionIndices();
	QVector< int >* getTextureCoordinateIndices();
	QVector< int >* getNormalIndices();

private:

	QVector< int > m_qvPositionIndices;
	QVector< int > m_qvTextureCoordinateIndices;
	QVector< int > m_qvNormalIndices;

	bool m_bHasTextureCoordinates;
	bool m_bHasNormals;

};

#endif // OBJ_FACE_H
