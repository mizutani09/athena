//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file user_table_class.cpp
//  \brief Implements class UserTable for an User-defined lookup table
//======================================================================================

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../inputs/ascii_table_reader.hpp"
#include "../inputs/hdf5_reader.hpp"
#include "../parameter_input.hpp"
#include "interp_table.hpp"

// // Order of datafields for HDF5 EOS tables
// const char *var_names[] = {"p/e(e/rho,rho)", "e/p(p/rho,rho)", "asq*rho/p(p/rho,rho)",
//                            "asq*rho/h(h/rho,rho)"};

// //----------------------------------------------------------------------------------------
// //! \fn void ReadBinaryTable(std::string fn, EosTable *peos_table)
// //  \brief Read data from binary EOS table and initialize interpolated table.
// void ReadBinaryTable(std::string fn, EosTable *peos_table) {
//   std::ifstream eos_file(fn.c_str(), std::ios::binary);
//   if (eos_file.is_open()) {
//     eos_file.seekg(0, std::ios::beg);
//     eos_file.read(reinterpret_cast<char*>(&peos_table->nVar), sizeof(peos_table->nVar));
//     eos_file.read(reinterpret_cast<char*>(&peos_table->nEgas), sizeof(peos_table->nEgas));
//     eos_file.read(reinterpret_cast<char*>(&peos_table->nRho), sizeof(peos_table->nRho));
//     eos_file.read(reinterpret_cast<char*>(&peos_table->logEgasMin),
//                   sizeof(peos_table->logEgasMin));
//     eos_file.read(reinterpret_cast<char*>(&peos_table->logEgasMax),
//                   sizeof(peos_table->logEgasMax));
//     eos_file.read(reinterpret_cast<char*>(&peos_table->logRhoMin),
//                   sizeof(peos_table->logRhoMin));
//     eos_file.read(reinterpret_cast<char*>(&peos_table->logRhoMax),
//                   sizeof(peos_table->logRhoMax));
//     peos_table->EosRatios.NewAthenaArray(peos_table->nVar);
//     eos_file.read(reinterpret_cast<char*>(peos_table->EosRatios.data()),
//                   peos_table->nVar * sizeof(peos_table->logRhoMin));
//     peos_table->table.SetSize(peos_table->nVar, peos_table->nEgas, peos_table->nRho);
//     peos_table->table.SetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
//     peos_table->table.SetX2lim(peos_table->logEgasMin, peos_table->logEgasMax);
//     eos_file.read(reinterpret_cast<char*>(peos_table->table.data.data()),
//                   peos_table->nVar * peos_table->nRho * peos_table->nEgas
//                   * sizeof(peos_table->logRhoMin));
//     eos_file.close();
//   } else {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in EosTable::EosTable, ReadBinaryTable" << std::endl
//         << "Unable to open eos table: " << fn << std::endl;
//     ATHENA_ERROR(msg);
//   }
// }

// //----------------------------------------------------------------------------------------
// //! \fn void ReadBinaryTable(std::string fn, EosTable *peos_table, ParameterInput *pin)
// //  \brief Read data from HDF5 EOS table and initialize interpolated table.
// void ReadHDF5Table(std::string fn, EosTable *peos_table, ParameterInput *pin) {
// #ifndef HDF5OUTPUT
//   {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in EosTable::EosTable, ReadHDF5Table" << std::endl
//         << "HDF5 EOS table specified, but HDF5 flag is not enabled."  << std::endl;
//     ATHENA_ERROR(msg);
//   }
// #endif
//   bool read_ratios = pin->GetOrAddBoolean("hydro", "eos_read_ratios", true);
//   std::string dens_lim_field =
//       pin->GetOrAddString("hydro", "EOS_dens_lim_field", "LogDensLim");
//   std::string espec_lim_field =
//       pin->GetOrAddString("hydro", "EOS_espec_lim_field", "LogEspecLim");
//   HDF5TableLoader(fn.c_str(), &peos_table->table, 4, var_names,
//                   espec_lim_field.c_str(), dens_lim_field.c_str());
//   peos_table->table.GetSize(peos_table->nVar, peos_table->nEgas, peos_table->nRho);
//   peos_table->table.GetX2lim(peos_table->logEgasMin, peos_table->logEgasMax);
//   peos_table->table.GetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
//   peos_table->EosRatios.NewAthenaArray(peos_table->nVar);
//   if (read_ratios) {
//     std::string ratio_field=pin->GetOrAddString("hydro", "EOS_ratio_field", "ratios");
//     int zero[] = {0};
//     int pnVar[] = {peos_table->nVar};
//     HDF5ReadRealArray(fn.c_str(), ratio_field.c_str(), 1, zero, pnVar,
//                       1, zero, pnVar, peos_table->EosRatios);
//     if (peos_table->EosRatios(0) <= 0) {
//       std::stringstream msg;
//       msg << "### FATAL ERROR in EosTable::EosTable, ReadHDF5Table" << std::endl
//           << "Invalid ratio. " << fn.c_str() << ", " << ratio_field << ", "
//           << peos_table->EosRatios(0) << std::endl;
//       ATHENA_ERROR(msg);
//     }
//   } else {
//     for (int i=0; i<peos_table->nVar; ++i) peos_table->EosRatios(i) = 1.0;
//   }
// }

//----------------------------------------------------------------------------------------
//! \fn void ReadAsciiTable(std::string fn, EosTable *peos_table, ParameterInput *pin)
//  \brief Read data from ascii table and initialize interpolated table.
void ReadAsciiTable(std::string tag, std::string fn, UserTable *puser_table, ParameterInput *pin) {
  bool read_ratios = pin->GetOrAddBoolean("problem", "usertab_"+tag+"_read_ratios", true);
  AthenaArray<Real> *pratios = nullptr;
  if (read_ratios) pratios = &puser_table->UserRatios;
  // If read_ratios then UserRatios.NewAthenaArray is called in ASCIITableLoader
  ASCIITableLoader(fn.c_str(), puser_table->table, pratios);
  puser_table->table.GetSize(puser_table->nVar, puser_table->nX2, puser_table->nX1);
  puser_table->table.GetX2lim(puser_table->X2Min, puser_table->X2Max);
  puser_table->table.GetX1lim(puser_table->X1Min, puser_table->X1Max);
  if (!read_ratios) {
    puser_table->UserRatios.NewAthenaArray(puser_table->nVar);
    for (int i=0; i<puser_table->nVar; ++i) puser_table->UserRatios(i) = 1.0;
  }
}

// UserTable constructor
UserTable::UserTable (ParameterInput *pin, std::string tag) {
  std::string user_fn, user_file_type;
  user_fn = pin->GetString("problem", "usertab_"+tag+"_file_name");
  user_file_type = pin->GetOrAddString("problem", "usertab_"+tag+"_file_type", "auto");
  // X1Unit = pin->GetOrAddReal("problem", "usertab_"+tag+"_X1_unit", 1.0);
  // X2Unit = pin->GetOrAddReal("problem", "usertab_"+tag+"_X2_unit", 1.0);
  //X3Unit = ...
  table = InterpTable2D();

  // Only Ascii file supported
  ReadAsciiTable(tag, user_fn, this, pin);
  
  // if (user_file_type.compare("auto") == 0) {
  //   std::string ext = eos_fn.substr(eos_fn.find_last_of(".") + 1);
  //   if (ext.compare("data")*ext.compare("bin") == 0) {
  //     eos_file_type.assign("binary");
  //   } else if (ext.compare("hdf5") == 0) {
  //     eos_file_type.assign("hdf5");
  //   } else if (ext.compare("tab")*ext.compare("txt")*ext.compare("ascii") == 0) {
  //     eos_file_type.assign("ascii");
  //   }
  // }
  // if (eos_file_type.compare("binary") == 0) { //Raw binary
  //   ReadBinaryTable(eos_fn, this);
  // } else if (eos_file_type.compare("hdf5") == 0) { // HDF5 table
  //   ReadHDF5Table(eos_fn, this, pin);
  // } else if (eos_file_type.compare("ascii") == 0) { // ASCII/text table
  //   ReadAsciiTable(eos_fn, this, pin);
  // } else {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in EosTable::EosTable" << std::endl
  //       << "EOS table of type '" << eos_file_type << "' not recognized."  << std::endl
  //       << "Options are 'ascii', 'binary', and 'hdf5'." << std::endl;
  //   ATHENA_ERROR(msg);
  // }
}

//----------------------------------------------------------------------------------------
//! \fn Real UserTable::GetUserTableRawData(int kOut, Real var2, Real var1,
//      Real var2Unit, Real var1Unit)
//  \brief Gets interpolated data from a table.
//   Note that normalization of input/output variables should be done outside this function.
Real UserTable::GetRawUserTableData(int kOut, Real x2, Real x1) {
  // not pow(10.,table.interpolate), because some variables are not in log10 scale.
  return table.interpolate(kOut, x2, x1);
}

// UserTable destructor
UserTable::~UserTable() {
  UserRatios.DeleteAthenaArray();
}
