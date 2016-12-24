#pragma once

#include <memory>
#include <string>
#include <vector>

class OBJData;
class OBJGroup;

class OBJLoader
{
public:

    static std::unique_ptr< OBJData > loadFile( const std::string& objFilename,
                             bool removeEmptyGroups = true );

private:

    static bool parseOBJ( const std::string& objFilename, OBJData* pOBJData );
    static bool parseMTL( const std::string& mtlFilename, OBJData* pOBJData );

    // parses a position line (starting with "v")
    // returns false on an error
    static bool parsePosition( int lineNumber, const std::string& line,
        const std::vector< std::string >& tokens, OBJData* pOBJData );

    // parses a texture coordinate line (starting with "vt")
    // returns false on an error
    static bool parseTextureCoordinate( int lineNumber, const std::string& line,
        const std::vector< std::string >& tokens, OBJData* pOBJData );

    // parses a normal line (starting with "vn")
    // returns false on an error
    static bool parseNormal( int lineNumber, const std::string& line,
        const std::vector< std::string >& tokens, OBJData* pOBJData );

    // parses a face line (starting with "f" or "fo")
    // returns false on an error
    static bool parseFace( int lineNumber, const std::string& line,
                          const std::vector< std::string >& tokens,
                          OBJGroup& currentGroup );

    // given the tokens in a face line
    // returns if the attached attributes are consistent
    static bool faceHasConsistentAttributes(
        const std::vector< std::string >& tokens,
        bool* pHasTextureCoordinates, bool* pHasNormals );

    // objFaceVertexToken is something of the form:
    // "int"
    // "int/int"
    // "int/int/int",
    // "int//int"
    // i.e. one of the delimited int strings that specify a vertex and its
    // attributes.
    //
    // Returns:
    // Whether the vertex is valid and the indices in the out parameters
    static bool getVertexAttributes( const std::string& objFaceVertexToken,
        int* pPositionIndex, int* pTextureCoordinateIndex, int* pNormalIndex );
};
