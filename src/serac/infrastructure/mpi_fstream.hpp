#pragma once

#include <fstream>
#include <mpi.h>

namespace mpi {

/// a tool for writing processor-specific log files
struct ofstream : public std::ofstream {

  /// open an output file for this processor (don't call directly)
  void initialize();

  /// @note don't call this before MPI_Init()
  template <typename T>
  friend ofstream& operator<<(ofstream&, T);

  ofstream() {
    initialized = false;
  }

  bool initialized;

};

/// analogous to operator<< used with e.g. std::cout
template <typename T>
ofstream& operator<<(ofstream& out, T op)
{
  if (!out.initialized) out.initialize();
  static_cast<std::ofstream&>(out) << op;
  return out;
}

/// the processor-specific output stream
extern ofstream out;

/// open an output file for this processor (don't call directly)
inline void ofstream::initialize() 
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  open("mpi_output_" + std::to_string(rank) + "_" + std::to_string(size) + ".txt", std::ios::out);
  initialized = true;
}

}  // namespace mpi
