#include "io/OBJLoader.h"

#include <fstream>
#include <pystring.h>

#include "io/OBJData.h"
#include "io/OBJGroup.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
std::unique_ptr< OBJData > OBJLoader::loadFile( const std::string& objFilename,
                                               bool removeEmptyGroups )
{
    std::unique_ptr< OBJData > pOBJData( new OBJData );
    bool succeeded = parseOBJ( objFilename, pOBJData.get() );
    if( !succeeded )
    {
        // Return null.
        pOBJData.reset();
    }

    if( removeEmptyGroups )
    {
        pOBJData->removeEmptyGroups();
    }

    return pOBJData;
}


//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

// static
bool OBJLoader::parseOBJ( const std::string& objFilename, OBJData* pOBJData )
{
    // Attempt to read the file.
    std::ifstream inputFile( objFilename );
    if (inputFile.fail())
    {
        return false;
    }

    int lineNumber = 0;
    std::string line = "";
    // Default group name is the empty string.
    OBJGroup& currentGroup = pOBJData->addGroup( "" );
    // Default material name is the empty string.
    pOBJData->addMaterial( "" );

    const std::string& delim( " " );
    while( std::getline( inputFile, line ) )
    {
        if( line == "" ||
           pystring::startswith( line, "#" ) ||
           pystring::startswith( line, "//" ) )
        {
            ++lineNumber;
            continue;
        }

        std::vector< std::string > tokens;
        pystring::split( line, tokens, delim );

        if( tokens.size() == 0 )
        {
            ++lineNumber;
            continue;
        }

        const std::string& commandToken = tokens[ 0 ];
        if( commandToken == "mtllib" )
        {
            const std::string& mtlRelativeFilename = tokens[ 1 ];
            std::string dirname = pystring::os::path::dirname( objFilename );
            std::string mtlFilename =
            pystring::os::path::join( dirname, mtlRelativeFilename );
            parseMTL( mtlFilename, pOBJData );
        }
        else if( commandToken == "g" )
        {
            std::string newGroupName;
            if( tokens.size() < 2 )
            {
                fprintf( stderr,
                        "Warning: group has no name, defaulting to ""\n"
                        "line %d: %s",
                        lineNumber, line.c_str() );

                newGroupName = "";
            }
            else
            {
                newGroupName = line.substr( 2 );
            }

            if( newGroupName != currentGroup.name() )
            {
                if( pOBJData->containsGroup( newGroupName ) )
                {
                    currentGroup = pOBJData->getGroupByName( newGroupName );
                }
                else
                {
                    currentGroup = pOBJData->addGroup( newGroupName );
                }
            }
        }
        else if( commandToken == "v" )
        {
            // TODO: error checking on all of these.
            OBJLoader::parsePosition( lineNumber, line, tokens, pOBJData );
        }
        else if( commandToken == "vt" )
        {
            OBJLoader::parseTextureCoordinate( lineNumber, line, tokens,
                                              pOBJData );
        }
        else if( commandToken == "vn" )
        {
            OBJLoader::parseNormal( lineNumber, line, tokens, pOBJData );
        }
        else if( commandToken == "usemtl" )
        {
            currentGroup.addMaterial( tokens[ 1 ] );
        }
        else if( commandToken == "f" || commandToken == "fo" )
        {
            OBJLoader::parseFace( lineNumber, line,
                tokens, currentGroup );
        }

        ++lineNumber;
    }

    return true;
}

// static
bool OBJLoader::parseMTL( const std::string& mtlFilename, OBJData* pOBJData )
{
    // Attempt to read the file.
    std::ifstream inputFile( mtlFilename );
    if (inputFile.fail())
    {
        return false;
    }

    int lineNumber = 0;
    std::string line;
    OBJMaterial& currentMaterial = pOBJData->getMaterialByName( "" );

    const std::string& delim( " " );
    while( std::getline( inputFile, line ) )
    {
        if( line == "" )
        {
            ++lineNumber;
            continue;
        }

        std::vector< std::string > tokens;
        pystring::split( line, tokens, delim );

        if( tokens.size() == 0 )
        {
            ++lineNumber;
            continue;
        }
        const std::string& commandToken = tokens[ 0 ];

        if( commandToken == "newmtl" )
        {
            std::string newMaterialName;

            if( tokens.size() < 2 )
            {
                fprintf( stderr,
                    "Warning: material has no name, defaulting to ""\n"
                    "line: %d\n%s",
                    lineNumber, line.c_str() );

                newMaterialName = "";
            }
            else
            {
                newMaterialName = tokens[ 1 ];
            }

            // if the new material's name isn't the same as the current one
            if( newMaterialName != currentMaterial.name() )
            {
                // but if it exists, then just set it as current
                if( pOBJData->containsMaterial( newMaterialName ) )
                {
                    currentMaterial = pOBJData->getMaterialByName(
                        newMaterialName );
                }
                // otherwise, make a new one and set it as current
                else
                {
                    currentMaterial = pOBJData->addMaterial( newMaterialName );
                }
            }
        }
        else if( commandToken == "Ka" )
        {
            // TODO: error checking, etc.
            float r = std::stof( tokens[ 1 ] );
            float g = std::stof( tokens[ 2 ] );
            float b = std::stof( tokens[ 3 ] );
            currentMaterial.setAmbientColor( { r, g, b } );
        }
        else if( commandToken == "Kd" )
        {
            float r = std::stof( tokens[ 1 ] );
            float g = std::stof( tokens[ 2 ] );
            float b = std::stof( tokens[ 3 ] );
            currentMaterial.setDiffuseColor( { r, g, b } );
        }
        else if( commandToken == "Ks" )
        {
            float r = std::stof( tokens[ 1 ] );
            float g = std::stof( tokens[ 2 ] );
            float b = std::stof( tokens[ 3 ] );
            currentMaterial.setSpecularColor( { r, g, b } );
        }
        else if( commandToken == "d" )
        {
            float d = std::stof( tokens[ 1 ] );
            currentMaterial.setAlpha( d );
        }
        else if( commandToken == "Ns" )
        {
            float ns = std::stof( tokens[ 1 ] );
            currentMaterial.setShininess( ns );
        }
        else if( commandToken == "illum" )
        {
            int il = std::stoi( tokens[ 1 ] );
            OBJMaterial::IlluminationModel illum =
                static_cast< OBJMaterial::IlluminationModel >( il );
            currentMaterial.setIlluminationModel( illum );
        }
        else if( commandToken == "map_Ka" )
        {
            if( tokens.size() > 1 )
            {
                currentMaterial.setAmbientTexture( tokens[ 1 ] );
            }
        }
        else if( commandToken == "map_Kd" )
        {
            if( tokens.size() > 1 )
            {
                currentMaterial.setDiffuseTexture( tokens[ 1 ] );
            }
        }

        ++lineNumber;
    }

    return true;
}

// static
bool OBJLoader::parsePosition( int lineNumber, const std::string& line,
                              const std::vector< std::string >& tokens,
                              OBJData* pOBJData )
{
    if( tokens.size() < 4 )
    {
        fprintf( stderr,
            "Incorrect number of tokens at line number: %d\n, %s\n",
            lineNumber, line.c_str() );
        return false;
    }
    else
    {
        // TODO: consider using std::stringstream
#if 0
        bool succeeded = true;

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
#endif

        float x = std::stof( tokens[ 1 ] );
        float y = std::stof( tokens[ 2 ] );
        float z = std::stof( tokens[ 3 ] );

        pOBJData->positions().push_back( { x, y, z } );

        return true;
    }
}

// static
bool OBJLoader::parseTextureCoordinate( int lineNumber, const std::string& line,
                                       const std::vector< std::string >& tokens,
                                       OBJData* pOBJData )
{
    if( tokens.size() < 3 )
    {
        fprintf( stderr,
            "Incorrect number of tokens at line number: %d\n, %s\n",
            lineNumber, line.c_str() );
        return false;
    }
    else
    {
#if 0
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
#endif

        float s = std::stof( tokens[ 1 ] );
        float t = std::stof( tokens[ 2 ] );

        pOBJData->textureCoordinates().push_back( Vector2f{ s, t } );

        return true;
    }
}

// static
bool OBJLoader::parseNormal( int lineNumber, const std::string& line,
                            const std::vector< std::string >& tokens,
                            OBJData* pOBJData )
{
    if( tokens.size() < 4 )
    {
        fprintf( stderr,
            "Incorrect number of tokens at line number: %d\n, %s\n",
            lineNumber, line.c_str() );
        return false;
    }
    else
    {
#if 0
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
#endif

        float nx = std::stof( tokens[ 1 ] );
        float ny = std::stof( tokens[ 2 ] );
        float nz = std::stof( tokens[ 3 ] );

        pOBJData->normals().push_back( { nx, ny, nz } );

        return true;
    }
}

// static
bool OBJLoader::parseFace( int lineNumber, const std::string& line,
                          const std::vector< std::string >& tokens,
                          OBJGroup& currentGroup )
{
    // TODO: support negative indices.
    if( tokens.size() < 4 )
    {
        fprintf( stderr,
                "Incorrect number of tokens at line number: %d\n, %s\n",
                lineNumber, line.c_str() );
        return false;
    }
    // first check line consistency - each vertex in the face
    // should have the same number of attributes
    bool faceIsValid;
    bool faceHasTextureCoordinates;
    bool faceHasNormals;

    faceIsValid = OBJLoader::faceHasConsistentAttributes( tokens,
        &faceHasTextureCoordinates, &faceHasNormals );

    if( !faceIsValid )
    {
        fprintf( stderr, "Face attributes inconsistent at line number: %d\n%s\n",
                lineNumber, line.c_str() );
        return false;
    }

    // ensure that all faces in a group are consistent:
    // they either all have texture coordinates or they don't
    // they either all have normals or they don't
    //
    // check how many faces the current group has
    // if the group has no faces, then the first face sets the group attributes

    if( currentGroup.numFaces() == 0 )
    {
        currentGroup.setHasTextureCoordinates( faceHasTextureCoordinates );
        currentGroup.setHasNormals( faceHasNormals );
    }

    bool faceIsConsistentWithGroup =
        ( currentGroup.hasTextureCoordinates() == faceHasTextureCoordinates )
        && ( currentGroup.hasNormals() == faceHasNormals );

    if( !faceIsConsistentWithGroup )
    {
        // TODO: boolToString()
        fprintf( stderr,
                "Face attributes inconsistent with group: %s at line: %d\n%s\n",
                currentGroup.name().c_str(), lineNumber, line.c_str() );
        fprintf( stderr, "group.hasTextureCoordinates() = %d\n",
                currentGroup.hasTextureCoordinates() );
        fprintf( stderr, "face.hasTextureCoordinates() = %d\n",
                faceHasTextureCoordinates );
        fprintf( stderr, "group.hasNormals() = %d\n",
                currentGroup.hasNormals() );
        fprintf( stderr, "face.hasNormals() = %d\n", faceHasNormals );

        return false;
    }

    OBJFace face( faceHasTextureCoordinates, faceHasNormals );

    // Process each vertex.
    for( int i = 1; i < tokens.size(); ++i )
    {
        int vertexPositionIndex;
        int vertexTextureCoordinateIndex;
        int vertexNormalIndex;

        OBJLoader::getVertexAttributes( tokens[ i ],
            &vertexPositionIndex, &vertexTextureCoordinateIndex,
            &vertexNormalIndex );

        face.positionIndices().push_back( vertexPositionIndex - 1 );

        if( faceHasTextureCoordinates )
        {
            face.textureCoordinateIndices().push_back(
                vertexTextureCoordinateIndex - 1 );
        }
        if( faceHasNormals )
        {
            face.normalIndices().push_back( vertexNormalIndex - 1 );
        }
    }

    currentGroup.addFace( face );
    return true;
}

// static
bool OBJLoader::faceHasConsistentAttributes(
    const std::vector< std::string >& tokens,
    bool* pHasTextureCoordinates, bool* pHasNormals )
{
    if( tokens.size() < 2 )
    {
        *pHasTextureCoordinates = false;
        *pHasNormals = false;
        return true;
    }

    int firstVertexPositionIndex;
    int firstVertexTextureCoordinateIndex;
    int firstVertexNormalIndex;

    bool firstVertexIsValid;
    bool firstVertexHasTextureCoordinates;
    bool firstVertexHasNormals;

    firstVertexIsValid = OBJLoader::getVertexAttributes( tokens[1],
        &firstVertexPositionIndex, &firstVertexTextureCoordinateIndex,
        &firstVertexNormalIndex );
    firstVertexHasTextureCoordinates =
        ( firstVertexTextureCoordinateIndex != -1 );
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
            &vertexPositionIndex, &vertexTextureCoordinateIndex,
            &vertexNormalIndex );
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
bool OBJLoader::getVertexAttributes( const std::string& objFaceVertexToken,
                                    int* pPositionIndex,
                                    int* pTextureCoordinateIndex,
                                    int* pNormalIndex )
{
    *pPositionIndex = 0;
    *pTextureCoordinateIndex = 0;
    *pNormalIndex = 0;

    std::vector< std::string > vertexAttributes;
    pystring::split( objFaceVertexToken, vertexAttributes, "/" );
    size_t numVertexAttributes = vertexAttributes.size();

    // Check if it has position. It is required.
    if( numVertexAttributes < 1 )
    {
        return false;
    }

    if( vertexAttributes[0] == "" )
    {
        return false;
    }

    // TODO: error checking on parsing the ints.
    *pPositionIndex = std::stoi( vertexAttributes[ 0 ] );
    if( numVertexAttributes > 1 )
    {
        if( vertexAttributes[1] != "" )
        {
            *pTextureCoordinateIndex =
                std::stoi( vertexAttributes[ 1 ] );
        }

        if( numVertexAttributes > 2 && vertexAttributes[2] != "" )
        {
            *pNormalIndex = std::stoi( vertexAttributes[ 2 ] );

        }
    }
    return true;
}
