#include <cmath>
#include <cstdio>
#include <string>
#include <sstream>

#include <gflags/gflags.h>
#include <third_party/pystring/pystring.h>

#include "libcgt/camera_wrappers/PoseStream.h"
#include "libcgt/core/io/File.h"
#include "libcgt/core/vecmath/EuclideanTransform.h"
#include "libcgt/core/vecmath/Quat4f.h"

using namespace libcgt::camera_wrappers;
using namespace libcgt::core;

DEFINE_string( input_file, "", "Input NVM file." );
DEFINE_string( output_file, "", "Output POSE file." );

// TODO: output pose format and direction
// TODO: whether to rotate the poses to GL
// TODO: rgbdtool should output a timestamp map.

using libcgt::camera_wrappers::PoseStreamMetadata;
using libcgt::camera_wrappers::PoseOutputStream;
using libcgt::camera_wrappers::PoseStreamTransformDirection;
using libcgt::core::vecmath::EuclideanTransform;

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );
    // TODO: validate input and output

    printf( "Input file: %s\n", FLAGS_input_file.c_str() );
    printf( "Output file: %s\n", FLAGS_output_file.c_str() );

    std::vector< std::string > nvm =
        File::readLines( FLAGS_input_file.c_str() );

    int nPoses = atoi( nvm[ 1 ].c_str() );
    std::vector< EuclideanTransform > outputPoses( nPoses );

    EuclideanTransform rot180(
        Matrix3f::rotateX(static_cast<float>( M_PI ) ) );

    // NVM's direction is camera_from_world.
    PoseStreamMetadata outputMetadata;
    outputMetadata.format = PoseStreamFormat::ROTATION_MATRIX_3X3_COL_MAJOR_AND_TRANSLATION_VECTOR_FLOAT;
    outputMetadata.units = PoseStreamUnits::ARBITRARY;
    outputMetadata.direction = PoseStreamTransformDirection::CAMERA_FROM_WORLD;

    PoseOutputStream outputStream( outputMetadata, FLAGS_output_file.c_str() );

    for( int i = 0; i < nPoses; ++i )
    {
        printf( "Writing pose %d of %d...\r", i, nPoses );
        const std::string& line = nvm[ 2 + i ];
        std::vector< std::string > tokens;
        pystring::split( line, tokens, " " );

        Quat4f q;
        Vector3f c;

        // Parse orientation.
        for( int j = 0; j < 4; ++j )
        {
            q[ j ] = static_cast< float >( atof( tokens[ 2 + j ].c_str() ) );
        }
        // Parse camera center.
        for( int j = 0; j < 3; ++j )
        {
            c[ j ] = static_cast< float >( atof( tokens[ 6 + j ].c_str() ) );
        }

        // Convert translation from SfM convention (R * (x[0:2] - x[3] * c)) to
        // standard convention: [R, t]: R * x[0:2] + t.
        EuclideanTransform e{ Matrix3f::fromQuat( q ), -q.rotateVector( c ) };

        // Convert from y-down (OpenCV convention) to y-up (OpenGL convention).
        EuclideanTransform gl_pose = rot180 * e * rot180;

        outputStream.write( i, i, gl_pose.rotation, gl_pose.translation );
    }
    printf( "done!\n" );

#if 0
    // How to read.
    libcgt::camera_wrappers::PoseInputStream inputStream(
        FLAGS_output_file.c_str() );
    auto metadata = inputStream.metadata();
    int32_t frameIndex;
    int64_t timestamp;
    EuclideanTransform e;
    bool ok = inputStream.read( frameIndex, timestamp, e.rotation, e.translation );
    while( ok )
    {
        ok = inputStream.read( frameIndex, timestamp, e.rotation, e.translation );
    }
#endif

    return 0;
}
