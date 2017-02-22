#pragma once

#include <string>

namespace libcgt { namespace qt_interop {

// Create all directories up to and including path.
// Returns true if succeeded (or if the directory already exists).
// Returns false otherwise.
//
// TODO(C++17): Remove this once C++17 is available with a file system api.
bool mkpath( const std::string& path );

} } // qt_interop, libcgt
