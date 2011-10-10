#include "io/OBJLoader.h"

#include <QFile>
#include <QStringList>
#include <QTextStream>
#include <QRegExp>

#include "io/OBJData.h"
#include "io/OBJGroup.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
OBJData* OBJLoader::loadFile( QString filename )
{
	// attempt to read the file
	// return NULL if failed
	QFile inputFile( filename );

	// try to open the file in read only mode
	if( !( inputFile.open( QIODevice::ReadOnly ) ) )
	{
		return NULL;
	}

	QTextStream inputTextStream( &inputFile );

	////////////////////////////////

	int lineNumber = 0;
	QString line = "";
	OBJData* pOBJData = new OBJData;
	OBJGroup* pCurrentGroup = pOBJData->addGroup( "" ); // default group name is the empty string
	
	QRegExp splitExp( "\\s+" );

	while( !( inputTextStream.atEnd() ) )
	{
		line = inputTextStream.readLine();
		++lineNumber;

		if( line != "" )
		{
			QStringList tokens = line.split( splitExp, QString::SkipEmptyParts );
			
			if( tokens.size() > 0 )
			{
				QString commandToken = tokens[ 0 ];

				if( commandToken == "g" )
				{
					QString newGroupName;

					if( tokens.size() < 2 )
					{
						fprintf( stderr, "Warning: group has no name, defaulting to ""\nline: %d\n%s",
							lineNumber, qPrintable( line ) );
						
						newGroupName = "";
					}
					else
					{
						newGroupName = tokens[ 1 ];
					}
					
					if( newGroupName != pCurrentGroup->getName() )
					{
						if( pOBJData->containsGroup( newGroupName ) )
						{
							pCurrentGroup = pOBJData->getGroup( newGroupName );
						}
						else
						{
							pCurrentGroup = pOBJData->addGroup( newGroupName );
						}						
					}
				}
				else if( commandToken == "v" )
				{
					// TODO: error checking on all of these and tear down pOBJData
					OBJLoader::parsePosition( lineNumber, line, tokens, pOBJData );
				}
				else if( commandToken == "vt" )
				{
					OBJLoader::parseTextureCoordinate( lineNumber, line, tokens, pOBJData );
				}
				else if( commandToken == "vn" )
				{
					OBJLoader::parseNormal( lineNumber, line, tokens, pOBJData );
				}
				else if( commandToken == "f" || commandToken == "fo" )
				{
					OBJLoader::parseFace( lineNumber, line,
						tokens, pCurrentGroup );
				}
			}
		}		
	}

	return pOBJData;
}



//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
bool OBJLoader::parsePosition( int lineNumber, QString line,
							  QStringList tokens, OBJData* pOBJData )
{
	if( tokens.size() < 4 )
	{
		fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
			lineNumber, qPrintable( line ) );
		return false;
	}
	else
	{
		bool succeeded;

		float x = tokens[ 1 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		float y = tokens[ 2 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		float z = tokens[ 3 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		pOBJData->getPositions()->append( Vector3f( x, y, z ) );

		return true;
	}
}

// static
bool OBJLoader::parseTextureCoordinate( int lineNumber, QString line,
									   QStringList tokens, OBJData* pOBJData )
{
	if( tokens.size() < 3 )
	{
		fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
			lineNumber, qPrintable( line ) );
		return false;
	}
	else
	{
		bool succeeded;

		float s = tokens[ 1 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		float t = tokens[ 2 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}		

		pOBJData->getTextureCoordinates()->append( Vector2f( s, t ) );

		return true;
	}
}

// static
bool OBJLoader::parseNormal( int lineNumber, QString line,
							QStringList tokens, OBJData* pOBJData )
{
	if( tokens.size() < 4 )
	{
		fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
			lineNumber, qPrintable( line ) );
		return false;
	}
	else
	{
		bool succeeded;

		float nx = tokens[ 1 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		float ny = tokens[ 2 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		float nz = tokens[ 3 ].toFloat( &succeeded );
		if( !succeeded )
		{
			fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		pOBJData->getNormals()->append( Vector3f( nx, ny, nz ) );

		return true;
	}
}

// static
bool OBJLoader::parseFace( int lineNumber, QString line,
						  QStringList tokens, OBJGroup* pCurrentGroup )
{
	if( tokens.size() < 4 )
	{
		fprintf( stderr, "Incorrect number of tokens at line number: %d\n, %s\n",
			lineNumber, qPrintable( line ) );
		return false;
	}
	else
	{
		// first check line consistency - each vertex in the face
		// should have the same number of attributes

		bool faceIsValid;
		bool faceHasTextureCoordinates;
		bool faceHasNormals;

		faceIsValid = OBJLoader::isFaceLineAttributesConsistent( tokens,
			&faceHasTextureCoordinates, &faceHasNormals );

		if( !faceIsValid )
		{
			fprintf( stderr, "Face attributes inconsistent at line number: %d\n%s\n",
				lineNumber, qPrintable( line ) );
			return false;
		}

		// ensure that all faces in a group are consistent
		// they either all have texture coordinates or they don't
		// they either all have normals or they don't
		// 
		// check how many faces the current group has
		// if the group has no faces, then the first vertex sets it

		if( pCurrentGroup->getFaces()->size() == 0 )
		{
			pCurrentGroup->setHasTextureCoordinates( faceHasTextureCoordinates );
			pCurrentGroup->setHasNormals( faceHasNormals );
		}

		bool faceIsConsistentWithGroup = ( pCurrentGroup->hasTextureCoordinates() == faceHasTextureCoordinates ) &&
			( pCurrentGroup->hasNormals() == faceHasNormals );

		if( !faceIsConsistentWithGroup )
		{
			fprintf( stderr, "Face attributes inconsistent with group: %s at line: %d\n%s\n",
				qPrintable( pCurrentGroup->getName() ), lineNumber, qPrintable( line ) );
			fprintf( stderr, "group.hasTextureCoordinates() = %d\n", pCurrentGroup->hasTextureCoordinates() );
			fprintf( stderr, "face.hasTextureCoordinates() = %d\n", faceHasTextureCoordinates );
			fprintf( stderr, "group.hasNormals() = %d\n", pCurrentGroup->hasNormals() );
			fprintf( stderr, "face.hasNormals() = %d\n", faceHasNormals );
			
			return false;
		}

		OBJFace face( faceHasTextureCoordinates, faceHasNormals );

		// for each vertex
		for( int i = 1; i < tokens.size(); ++i )
		{
			int vertexPositionIndex;
			int vertexTextureCoordinateIndex;
			int vertexNormalIndex;

			OBJLoader::getVertexAttributes( tokens[ i ],
				&vertexPositionIndex, &vertexTextureCoordinateIndex, &vertexNormalIndex );

			face.getPositionIndices()->append( vertexPositionIndex );

			if( faceHasTextureCoordinates )
			{
				face.getTextureCoordinateIndices()->append( vertexTextureCoordinateIndex );
			}
			if( faceHasNormals )
			{
				face.getNormalIndices()->append( vertexNormalIndex );
			}
		}

		pCurrentGroup->getFaces()->append( face );
		return true;
	}
}

// static
bool OBJLoader::isFaceLineAttributesConsistent( QStringList tokens,
											   bool* pHasTextureCoordinates, bool* pHasNormals )
{
	int firstVertexPositionIndex;
	int firstVertexTextureCoordinateIndex;
	int firstVertexNormalIndex;

	bool firstVertexIsValid;
	bool firstVertexHasTextureCoordinates;
	bool firstVertexHasNormals;

	firstVertexIsValid = OBJLoader::getVertexAttributes( tokens[1],
		&firstVertexPositionIndex, &firstVertexTextureCoordinateIndex, &firstVertexNormalIndex );
	firstVertexHasTextureCoordinates = ( firstVertexTextureCoordinateIndex != -1 );
	firstVertexHasNormals = ( firstVertexNormalIndex != -1 );

	if( !firstVertexIsValid )
	{
		*pHasTextureCoordinates = false;
		*pHasNormals = false;
		return false;
	}

	for( int i = 2; i < tokens.size(); ++i )
	{
		int vertexPositionIndex;
		int vertexTextureCoordinateIndex;
		int vertexNormalIndex;

		bool vertexIsValid;
		bool vertexHasTextureCoordinates;
		bool vertexHasNormals;

		vertexIsValid = OBJLoader::getVertexAttributes( tokens[i],
			&vertexPositionIndex, &vertexTextureCoordinateIndex, &vertexNormalIndex );
		vertexHasTextureCoordinates = ( vertexTextureCoordinateIndex != -1 );
		vertexHasNormals = ( vertexNormalIndex != -1 );

		if( !vertexIsValid )
		{
			*pHasTextureCoordinates = false;
			*pHasNormals = false;
			return false;
		}

		if( firstVertexHasTextureCoordinates != vertexHasTextureCoordinates )
		{
			*pHasTextureCoordinates = false;
			*pHasNormals = false;
			return false;
		}

		if( firstVertexHasNormals != vertexHasNormals )
		{
			*pHasTextureCoordinates = false;
			*pHasNormals = false;
			return false;
		}
	}

	*pHasTextureCoordinates = firstVertexHasTextureCoordinates;
	*pHasNormals = firstVertexHasNormals;
	return true;
}

// static
bool OBJLoader::getVertexAttributes( QString objFaceVertexToken,
									int* pPositionIndex, int* pTextureCoordinateIndex, int* pNormalIndex )
{
	QStringList vertexAttributes = objFaceVertexToken.split( "/", QString::KeepEmptyParts );
	int vertexNumAttributes = vertexAttributes.size();

	// check if it has position
	if( vertexNumAttributes < 1 )
	{
		*pPositionIndex = -1;
		*pTextureCoordinateIndex = -1;
		*pNormalIndex = -1;
		return false;
	}
	else
	{
		if( vertexAttributes[0] == "" )
		{
			*pPositionIndex = -1;
			*pTextureCoordinateIndex = -1;
			*pNormalIndex = -1;
			return false;
		}
		else
		{
			// TODO: error checking on parsing the ints?
			*pPositionIndex = vertexAttributes[ 0 ].toInt() - 1;

			if( vertexNumAttributes > 1 )
			{
				if( vertexAttributes[1] == "" )
				{
					*pTextureCoordinateIndex = -1;
				}
				else
				{
					*pTextureCoordinateIndex = vertexAttributes[ 1 ].toInt() - 1;
				}

				if( vertexNumAttributes > 2 )
				{
					if( vertexAttributes[2] == "" )
					{
						*pNormalIndex = -1;
					}
					else
					{
						*pNormalIndex = vertexAttributes[ 2 ].toInt() - 1;
					}
				}
				else
				{
					*pNormalIndex = -1;
				}				
			}
			else
			{
				*pTextureCoordinateIndex = -1;
				*pNormalIndex = -1;
			}
		}

		return true;
	}
}
