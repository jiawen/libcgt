#pragma once

#include <memory>
#include <vector>
#include <map>

#include <common/Comparators.h>

#include "io/OBJData.h"
#include "io/OBJGroup.h"
#include "io/OBJFace.h"

#include "vecmath/Vector3f.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"

class TriangleMesh
{
public:

	TriangleMesh();
	TriangleMesh( std::shared_ptr< OBJData > pData );
	TriangleMesh( std::shared_ptr< OBJData > pData, int groupIndex );

	float area( int faceIndex ) const;
	float totalArea() const;

	// given a set of connected faces in connectedComponent
	// returns a consolidated mesh
	// where vertices not referenced by a face are removed
	// and faces index vertices from [0,nVertices)
	TriangleMesh consolidate( const std::vector< int >& connectedComponent );

	// returns the number of pruned faces
	// if it's 0, then edgeToFace is valid
	// replaces m_faces with a set of valid faces
	//int pruneInvalidFaces( QHash< Vector2i, int >& edgeToFace );
	int pruneInvalidFaces( std::map< Vector2i, int >& edgeToFace );
	void buildAdjacency();

	void computeConnectedComponents();

	void computeAreas();

	std::vector< Vector3f > m_positions;
	std::vector< Vector3f > m_normals;
	std::vector< Vector3i > m_faces;

	std::map< Vector2i, int > m_edgeToFace;
	std::vector< std::vector< int > > m_faceToFace;

	// connected components of faces sharing an edge
	// m_connectedComponents.size() is the number of components
	// each m_connectedComponents[i] is a vector of face indices
	// belonging to that component
	std::vector< std::vector< int > > m_connectedComponents;

	std::vector< float > m_areas;
};
