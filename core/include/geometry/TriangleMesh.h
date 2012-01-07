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

	TriangleMesh( std::shared_ptr< OBJData > pData )
	{
		QVector< Vector3f >* pPositions = pData->getPositions();
		int nVertices = pPositions->size();
		m_positions = std::vector< Vector3f >( nVertices );

		for( int v = 0; v < pPositions->size(); ++v )
		{
			m_positions[ v ] = pPositions->at( v );
		}

		QVector< OBJGroup* >* pGroups = pData->getGroups();
		for( int g = 0; g < pGroups->size(); ++g )
		{
			auto pGroup = pGroups->at( g );
			auto pFaces = pGroup->getFaces();
			for( int f = 0; f < pFaces->size(); ++f )
			{
				auto pFace = pFaces->at( f );
				int i0 = pFace.getPositionIndices()->at( 0 );
				int i1 = pFace.getPositionIndices()->at( 1 );
				int i2 = pFace.getPositionIndices()->at( 2 );

				m_faces.push_back( Vector3i( i0, i1, i2 ) );
			}
		}
	}

	// returns the number of pruned faces
	// if it's 0, then edgeToFace is valid
	// replaces m_faces with a set of valid faces
	//int pruneInvalidFaces( QHash< Vector2i, int >& edgeToFace );
	int pruneInvalidFaces( std::map< Vector2i, int >& edgeToFace );
	void buildAdjacency();

	void connectedComponents();

	std::vector< Vector3f > m_positions;
	std::vector< Vector3i > m_faces;

	//QHash< Vector2i, int > m_edgeToFace;
	std::map< Vector2i, int > m_edgeToFace;
	std::vector< std::vector< int > > m_faceToFace;

	// connected components of faces sharing an edge
	// m_connectedComponents.size() is the number of components
	// each m_connectedComponents[i] is a vector of face indices
	// belonging to that component
	std::vector< std::vector< int > > m_connectedComponents;
};