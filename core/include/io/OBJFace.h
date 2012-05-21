#pragma once

#include <vector>

class OBJFace
{
public:

	OBJFace();
	OBJFace( bool hasTextureCoordinates, bool hasNormals );

	bool hasTextureCoordinates() const;
	bool hasNormals() const;

	int numVertices() const;

	const std::vector< int >& positionIndices() const;
	std::vector< int >& positionIndices();

	const std::vector< int >& textureCoordinateIndices() const;
	std::vector< int >& textureCoordinateIndices();

	const std::vector< int >& normalIndices() const;
	std::vector< int >& normalIndices();

private:

	std::vector< int > m_positionIndices;
	std::vector< int > m_textureCoordinateIndices;
	std::vector< int > m_normalIndices;

	bool m_bHasTextureCoordinates;
	bool m_bHasNormals;

};
