#pragma once

#include <memory>
#include <vector>
#include <map>
#include <list>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/Comparators.h"
#include "libcgt/core/io/OBJData.h"
#include "libcgt/core/io/OBJGroup.h"
#include "libcgt/core/io/OBJFace.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector3i.h"

class TriangleMesh
{
public:

    TriangleMesh() = default;

    // Construct a triangle mesh out of a list of positions and
    // and a list of triplets indexing into it.
    TriangleMesh( Array1DReadView< Vector3f > positions,
        Array1DReadView< Vector3i > faces );

    // Construct a triangle mesh out of a list of positions and normals,
    // and a list of triplets indexing into them.
    // "positions" and "normals" must have the same size.
    TriangleMesh( Array1DReadView< Vector3f > positions,
        Array1DReadView< Vector3f > normals,
        Array1DReadView< Vector3i > faces );

    // make a triangle mesh out of data from an OBJ file
    // all groups are merged such that:
    //   all positions are in one array
    //   all normals are in one array, consolidated to be the same size as positions
    //      (extra normals are discarded)
    //   all faces are triangulated and re-indexed to point to the same array
    // TODO: generate normals per face (need a separate array to keep track of additional normals)
    TriangleMesh( const OBJData& data );

    // make a triangle mesh out of data from an OBJ file
    // (one particular group)
    // TODO: also triangulate face
    TriangleMesh( const OBJData& data, int groupIndex,
        bool generatePerFaceNormalsIfNonExistent = true );

    int numVertices() const;
    int numFaces() const;

    const std::vector< Vector3f >& positions() const;
    std::vector< Vector3f >& positions();

    const std::vector< Vector3f >& normals() const;
    std::vector< Vector3f >& normals();

    const std::vector< Vector3i >& faces() const;
    std::vector< Vector3i >& faces();

    // returns -1 if edge i --> j is not on a face
    int vertexOppositeEdge( int i, int j ) const;

    // returns -1 if edge i --> j is not on a face
    int vertexOppositeEdge( const Vector2i& ij ) const;

    float meanEdgeLength() const;

    float area( int faceIndex ) const;
    float totalArea() const;

    bool obtuse( int faceIndex ) const;

    bool intersectRay( const Vector3f& origin, const Vector3f& direction,
        float& t, Vector3f& barycentrics, int& faceIndex,
        float tMin = 0 ) const;

    Vector3f barycentricInterpolatePosition( int faceIndex, const Vector3f& barycentrics ) const;

    // returns a normalized normal
    Vector3f barycentricInterpolateNormal( int faceIndex, const Vector3f& barycentrics ) const;

    // given a set of connected faces in connectedComponent
    // returns a consolidated mesh
    // where vertices not referenced by a face are removed
    // and faces index vertices from [0,nVertices)
    TriangleMesh consolidate( const std::vector< int >& connectedComponent );

    // checks that each edge is shared by at most 2 triangles (1 in each direction)
    // if an edge (v0,v1) is touched more than once, the second face is discarded
    //   (TODO: split the edge?)
    //
    // returns the number of pruned faces
    // if it's 0, then edgeToFace is valid
    // replaces m_faces with a set of valid faces
    int pruneInvalidFaces( std::map< Vector2i, int >& edgeToFace );

    void buildAdjacency();
    void invalidateAdjancency();


    void computeConnectedComponents();

    void computeAreas();

    void computeEdgeLengths();

    // e = (v0,v1) is a boundary edge if it there is no twin edge (v1,v0)
    bool isBoundaryEdge( const Vector2i& edge );

    std::vector< Vector3f > m_positions;
    std::vector< Vector3f > m_normals;

    // each face indexes into m_positions
    std::vector< Vector3i > m_faces;

    // connected components of faces sharing an edge
    // m_connectedComponents.size() is the number of components
    // each m_connectedComponents[i] is a vector of face indices
    // belonging to that component
    std::vector< std::vector< int > > m_connectedComponents;

    std::vector< float > m_areas;

    std::map< Vector2i, float > m_edgeLengths;

    bool saveOBJ( const std::string& filename );

    // TODO: mark if one-ring is closed?
    // or use an actual linked list
    std::vector< bool > m_oneRingIsClosed; // whether the one ring is closed (loops around)
    std::vector< std::list< int > > m_vertexToVertex; // vertex index --> one-ring vertices, in counterclockwise order
    std::vector< std::list< int > > m_vertexToFace; // vertex index --> one-ring faces, in counterclockwise order

    // adjacency data structures
    std::map< Vector2i, int > m_edgeToFace; // edge (v0,v1) --> face index

    // TODO: use a Vector3i, use -1 to indicate missing face
    std::vector< std::vector< int > > m_faceToFace; // face index --> adjacent face indices (shared by edges, not vertices, at most 3, and in counterclockwise order)

    std::map< Vector2i, Vector2i > m_edgeToPrevEdge;
    std::map< Vector2i, Vector2i > m_edgeToNextEdge;

    std::vector< int > m_vertexToOutgoingEdge; // vertex index --> one outgoing edge

private:

    // Marks whether the cached adjacency data structures are dirty.
    bool m_adjacencyIsDirty = true;

    // TODO(jiawen): make this a pure function, not a method.
    // the input data might have different number of normals vs vertices
    // if the input has *more* normals,
    //   then some of them are clearly unused and can be pruned
    // if the input has *fewer* normals,
    //   then some of them are shared and should be duplicated
    // they indices should line up with the positions
    // since the faces indexing them is authoritative
    void consolidateNormalsWithPositions( const std::vector< Vector3i >& normalIndices );
};
