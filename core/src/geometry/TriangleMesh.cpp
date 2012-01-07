#include "geometry/TriangleMesh.h"

#include "common/ProgressReporter.h"

#include <algorithm>
#include <stack>

//int TriangleMesh::pruneInvalidFaces( QHash< Vector2i, int >& edgeToFace )
int TriangleMesh::pruneInvalidFaces( std::map< Vector2i, int >& edgeToFace )
{
	// walk over all faces
	// build for each edge (v0,v1)
	//   edgeToFace[ v0, v1 ] = face
	//   if it already exists, we have a problem
	//   and we will throw the face away

	int nFaces = m_faces.size();
	edgeToFace.clear();
	//edgeToFace.reserve( 3 * nFaces );
	std::vector< Vector3i > validFaces;
	validFaces.reserve( nFaces );

	ProgressReporter pr( "Pruning invalid faces", nFaces );

	int nPruned = 0;
	for( int f = 0; f < nFaces; ++f )
	{
		Vector3i face = m_faces[ f ];

		Vector2i e0 = face.xy();
		Vector2i e1 = face.yz();
		Vector2i e2 = face.zx();

		/*
		if( !edgeToFace.contains( e0 ) &&
			!edgeToFace.contains( e1 ) &&
			!edgeToFace.contains( e2 ) )
		*/
		if( edgeToFace.find( e0 ) == edgeToFace.end() &&
			edgeToFace.find( e1 ) == edgeToFace.end() &&
			edgeToFace.find( e2 ) == edgeToFace.end() )
		{
			edgeToFace[ e0 ] = f;
			edgeToFace[ e1 ] = f;
			edgeToFace[ e2 ] = f;
			validFaces.push_back( face );
		}
		else
		{
			++nPruned;

			fprintf( stderr, "Found invalid face: (%d, %d, %d)\n",
				face.x, face.y, face.z );
			
#if 0
			if( edgeToFace.contains( e0 ) )
			{
				Vector3i existingFace = m_faces[ edgeToFace.value( e0 ) ];
				fprintf( stderr, "Existing face: (%d, %d, %d)\n",
					existingFace.x, existingFace.y, existingFace.z );
			}
			if( edgeToFace.contains( e1 ) )
			{
				Vector3i existingFace = m_faces[ edgeToFace.value( e1 ) ];
				fprintf( stderr, "Existing face: (%d, %d, %d)\n",
					existingFace.x, existingFace.y, existingFace.z );
			}
			if( edgeToFace.contains( e2 ) )
			{
				Vector3i existingFace = m_faces[ edgeToFace.value( e2 ) ];
				fprintf( stderr, "Existing face: (%d, %d, %d)\n",
					existingFace.x, existingFace.y, existingFace.z );
			}
#endif
		}

		pr.notifyAndPrintProgressString();
	}
	if( nPruned > 0 )
	{
		fprintf( stderr, "Pruned %d faces\n", nPruned );
		m_faces = validFaces;
	}

	return nPruned;
}

void TriangleMesh::buildAdjacency()
{
	printf( "pruning invalid faces\n" );
	int nPruned = pruneInvalidFaces( m_edgeToFace );

	// walk over all faces
	// and build an adjacency map:
	// edge -> adjacent face
	if( nPruned != 0 )
	{
		int nFaces = m_faces.size();
		m_edgeToFace.clear();
		//m_edgeToFace.reserve( 3 * nFaces );

		for( int f = 0; f < nFaces; ++f )
		{
			Vector3i face = m_faces[ f ];

			Vector2i e0 = face.xy();
			Vector2i e1 = face.yz();
			Vector2i e2 = face.zx();

			m_edgeToFace[ e0 ] = f;
			m_edgeToFace[ e1 ] = f;
			m_edgeToFace[ e2 ] = f;
		}
	}

	// build face to face adjacency
	// for each face:
	//    find 3 edges
	//    flip edge: if the flipped edge has an adjacent face
	//       add it as a neighbor of this face
	printf( "building face to face adjacency\n" );
	int nFaces = m_faces.size();
	m_faceToFace.clear();
	m_faceToFace.resize( nFaces );
	for( int f = 0; f < nFaces; ++f )
	{
		Vector3i face = m_faces[ f ];
		// get 3 edge twins
		Vector2i e0t = face.yx();
		Vector2i e1t = face.zy();
		Vector2i e2t = face.xz();

		//if( m_edgeToFace.contains( e0t ) )
		if( m_edgeToFace.find( e0t ) != m_edgeToFace.end() )
		{
			m_faceToFace[ f ].push_back( m_edgeToFace[ e0t ] );
		}
		//if( m_edgeToFace.contains( e1t ) )
		if( m_edgeToFace.find( e1t ) != m_edgeToFace.end() )
		{
			m_faceToFace[ f ].push_back( m_edgeToFace[ e1t ] );
		}
		//if( m_edgeToFace.contains( e2t ) )
		if( m_edgeToFace.find( e2t ) != m_edgeToFace.end() )
		{
			m_faceToFace[ f ].push_back( m_edgeToFace[ e2t ] );
		}
	}
}

void TriangleMesh::connectedComponents()
{
	m_connectedComponents.clear();

	// build a bit vector of length nFaces
	// set them all to true for now
	int nFaces = m_faces.size();
	std::vector< bool > remainingFaces( nFaces, true );

	// debugging
	int nComponents = 0;
	int nProcessedFaces = 0;

	// loop until out of faces
	auto rootItr = std::find( remainingFaces.begin(), remainingFaces.end(), true );
	while( rootItr != remainingFaces.end() )
	{
		int currentFaceIndex = static_cast< int >( rootItr - remainingFaces.begin() );
		std::vector< int > connectedComponent;
		
		// start with a root face and push it onto the stack
		// while the stack is not empty
		//    pop a face off the stack and add it to the component
		//    mark it as taken
		//    then add its adjacent faces onto the stack
		std::stack< int > adjStack;
		adjStack.push( currentFaceIndex );
		remainingFaces[ currentFaceIndex ] = false;
		while( !( adjStack.empty() ) )
		{
			int currentFaceIndex = adjStack.top();
			adjStack.pop();
			
			connectedComponent.push_back( currentFaceIndex );

			for( int i = 0; i < m_faceToFace[ currentFaceIndex ].size(); ++i )
			{
				int adjacentFaceIndex = m_faceToFace[ currentFaceIndex ][ i ];
				if( remainingFaces[ adjacentFaceIndex ] )
				{
					adjStack.push( adjacentFaceIndex );
					remainingFaces[ adjacentFaceIndex ] = false;
				}
			}
		}

		++nComponents;
		int componentSize = connectedComponent.size();
		nProcessedFaces += componentSize;
		printf( "Found %d components so far, this component %d faces, %d total faces processed, %d faces total.\n",
			nComponents, componentSize, nProcessedFaces, nFaces );

		m_connectedComponents.push_back( connectedComponent );
		rootItr = std::find( remainingFaces.begin(), remainingFaces.end(), true );
	}

	fprintf( stderr, "Found %d connected components\n", m_connectedComponents.size() );
}
