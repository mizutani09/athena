//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ascii_table_reader.cpp
//! \brief Implements ASCII table reader functions

// C headers

// C++ headers
#include <fstream>
#include <iostream>   // ifstream
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../athena.hpp"             // Real
#include "../athena_arrays.hpp"      // AthenaArray
#include "../utils/interp_table.hpp" // InterpTable2D
#include "ascii_table_reader.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ASCIITableLoader(const char *filename, InterpTable2D* ptable,
//!                           AthenaArray<Real>* pratios)
//! \brief Load a table stored in ASCII form and initialize an interpolated table.
//!        Fastest index corresponds to column
void ASCIITableLoader(const char *filename, InterpTable2D &table,
                      AthenaArray<Real>* pratios) {
  std::ifstream file(filename, std::ios::in);
  std::string line;

  if (!file.is_open()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "Failed to open EOS table file: " << filename << std::endl;
    ATHENA_ERROR(msg);
  }

  auto read_data_line = [&](const char* what) {
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') continue;
      return;
    }
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "Unexpected EOF while reading " << what
        << " from file: " << filename << std::endl;
    ATHENA_ERROR(msg);
  };

  int nvar = 0, nx2 = 0, nx1 = 0;
  read_data_line("table shape");
  std::stringstream stream(line);
  if (!(stream >> nvar >> nx2 >> nx1)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "Failed to parse table shape line: \"" << line << "\"" << std::endl
        << "file: " << filename << std::endl;
    ATHENA_ERROR(msg);
  }
  if (nvar < 1 || nx2 < 2 || nx1 < 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "Invalid shape: (" << nvar << ", " << nx2 << ", " << nx1 << ")" << std::endl
        << "file: " << filename << std::endl;
    ATHENA_ERROR(msg);
  }
  table.SetSize(nvar, nx2, nx1);

  Real min_, max_;
  read_data_line("x2 limits");
  stream.clear();
  stream.str(line);
  if (!(stream >> min_ >> max_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "Failed to parse x2 limits: \"" << line << "\"" << std::endl
        << "file: " << filename << std::endl;
    ATHENA_ERROR(msg);
  }
  if (min_ >= max_) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "x2min>=x2max." << std::endl;
    ATHENA_ERROR(msg);
  }
  table.SetX2lim(min_, max_);

  read_data_line("x1 limits");
  stream.clear();
  stream.str(line);
  if (!(stream >> min_ >> max_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "Failed to parse x1 limits: \"" << line << "\"" << std::endl
        << "file: " << filename << std::endl;
    ATHENA_ERROR(msg);
  }
  if (min_ >= max_) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
        << "x1min>=x1max." << std::endl;
    ATHENA_ERROR(msg);
  }
  table.SetX1lim(min_, max_);

  if (pratios != nullptr) {
    read_data_line("ratios");
    stream.clear();
    stream.str(line);
    pratios->NewAthenaArray(nvar);
    for (int i = 0; i < nvar; ++i) {
      if (!(stream >> (*pratios)(i))) {
        std::stringstream msg;
        msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
            << "Failed to parse ratio[" << i << "] from line: \"" << line << "\"" << std::endl
            << "file: " << filename << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }

  for (int row = 0; row < nx2 * nvar; ++row) {
    read_data_line("table data");
    std::stringstream lstream(line);
    for (int col = 0; col < nx1; ++col) {
      if (!(lstream >> table.data(row, col))) {
        std::stringstream msg;
        msg << "### FATAL ERROR in ASCIITableLoader" << std::endl
            << "Failed to parse table value at row=" << row << ", col=" << col << std::endl
            << "line: \"" << line << "\"" << std::endl
            << "file: " << filename << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }
}
