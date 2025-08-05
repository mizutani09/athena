//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file read_opacity_table.cpp
//  \brief Implements class UserOpacityTable for an User-defined lookup table
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
#include "../utils/interp_table.hpp"
#include "rad_fld.hpp"

#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

// Order of datafields for HDF5 opacity tables
// These variable names should match the dataset names in your HDF5 opacity file
// Order must match enum OpacityIndex {SIGMA_P=0, SIGMA_R=1} in rad_fld.hpp
const char *opacity_var_names[] = {"planck_mean_opacity", "rosseland_mean_opacity"};

//----------------------------------------------------------------------------------------
//! \fn void ReadAsciiOpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin)
//  \brief Read data from ascii opacity table and initialize interpolated table.
void ReadAsciiOpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin) {
  bool read_tables = pin->GetOrAddBoolean("problem", "opacity_table_read_tables", true);
  AthenaArray<Real> *ptables = nullptr;
  if (read_tables) ptables = &puser_table->OpacityTables;

  // If read_tables then OpacityTables.NewAthenaArray is called in ASCIITableLoader
  ASCIITableLoader(fn.c_str(), *puser_table, ptables);
  puser_table->GetSize(puser_table->nVar, puser_table->nDensity, puser_table->nTemp);
  puser_table->GetX2lim(puser_table->densityMin, puser_table->densityMax);
  puser_table->GetX1lim(puser_table->tempMin, puser_table->tempMax);

  // Apply unit conversion from physical units (cm^2/g) to code units
  for (int ivar = 0; ivar < puser_table->nVar; ++ivar) {
    for (int j = 0; j < puser_table->nDensity; ++j) {
      for (int i = 0; i < puser_table->nTemp; ++i) {
        puser_table->data(ivar, j, i) /= puser_table->opacity_unit;
      }
    }
  }

  if (!read_tables) {
    puser_table->OpacityTables.NewAthenaArray(puser_table->nVar);
    for (int i=0; i<puser_table->nVar; ++i) puser_table->OpacityTables(i) = 1.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ReadHDF5OpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin)
//  \brief Read data from HDF5 opacity table and initialize interpolated table.
void ReadHDF5OpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin) {
  #ifdef HDF5OUTPUT
  // Get number of variables to read from the table
  int nvar = RadFLD::NOPACITY;

  // Get variable names - these should match the dataset names in your HDF5 file
  const char **var_names = opacity_var_names;

  // For this opacity table format, we have 1D arrays that need to be read separately
  // First, read the coordinate arrays to get dimensions
  AthenaArray<Real> temp_array, density_array;

  // Read temperature coordinate array first to get its size
  int temp_size = 0;
  {
    hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
    hid_t dataset = H5Dopen(file, "temperature", H5P_DEFAULT);
    hid_t dspace = H5Dget_space(dataset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    temp_size = static_cast<int>(dims[0]);
    H5Sclose(dspace);
    H5Dclose(dataset);
    H5Fclose(file);
    H5Pclose(property_list_file);
  }

  // Read density coordinate array to get its size
  int density_size = 0;
  {
    hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
    hid_t dataset = H5Dopen(file, "density", H5P_DEFAULT);
    hid_t dspace = H5Dget_space(dataset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    density_size = static_cast<int>(dims[0]);
    H5Sclose(dspace);
    H5Dclose(dataset);
    H5Fclose(file);
    H5Pclose(property_list_file);
  }

  // Assuming the opacity data is organized as a 1D array that corresponds to
  // a flattened 2D grid where coordinates are paired (T[i], rho[i] -> opacity[i])
  int nx1 = temp_size;   // temperature dimension
  int nx2 = density_size;  // density dimension

  // Check if we have the same number of temperature and density points
  if (temp_size != density_size) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Temperature and density arrays must have the same size." << std::endl
        << "temp_size = " << temp_size << ", density_size = " << density_size << std::endl;
    ATHENA_ERROR(msg);
  }

  // Set up the table structure - for 1D paired data, we treat it as a 1D table
  // but InterpTable2D requires 2D, so we'll create a minimal 2D structure
  puser_table->SetSize(nvar, 1, nx1);  // nx2=1, nx1=temp_size
  puser_table->nVar = nvar;
  puser_table->nTemp = nx1;
  puser_table->nDensity = 1;  // Since data is 1D paired

  // Read coordinate arrays
  temp_array.NewAthenaArray(temp_size);
  density_array.NewAthenaArray(density_size);

  int start_file[1] = {0};
  int start_mem[1] = {0};
  int count_temp[1] = {temp_size};
  int count_density[1] = {density_size};

  HDF5ReadRealArray(fn.c_str(), "temperature", 1, start_file, count_temp,
                    1, start_mem, count_temp, temp_array);
  HDF5ReadRealArray(fn.c_str(), "density", 1, start_file, count_density,
                    1, start_mem, count_density, density_array);

  // Set coordinate limits
  puser_table->tempMin = temp_array(0);
  puser_table->tempMax = temp_array(temp_size - 1);
  puser_table->densityMin = density_array(0);
  puser_table->densityMax = density_array(density_size - 1);

  puser_table->SetX1lim(puser_table->tempMin, puser_table->tempMax);
  puser_table->SetX2lim(puser_table->densityMin, puser_table->densityMax);

  // Store coordinate arrays for 1D paired data interpolation
  puser_table->temp_coords.NewAthenaArray(temp_size);
  puser_table->density_coords.NewAthenaArray(density_size);
  for (int i = 0; i < temp_size; ++i) {
    puser_table->temp_coords(i) = temp_array(i)/puser_table->T_unit; // Convert to code units
    puser_table->density_coords(i) = density_array(i)/puser_table->rho_unit; // Convert to code units
  }

  // Read opacity data arrays
  for (int ivar = 0; ivar < nvar; ++ivar) {
    AthenaArray<Real> opacity_1d;

    // Get opacity array size
    int opacity_size = 0;
    {
      hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
      hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
      hid_t dataset = H5Dopen(file, var_names[ivar], H5P_DEFAULT);
      hid_t dspace = H5Dget_space(dataset);
      hsize_t dims[1];
      H5Sget_simple_extent_dims(dspace, dims, NULL);
      opacity_size = static_cast<int>(dims[0]);
      H5Sclose(dspace);
      H5Dclose(dataset);
      H5Fclose(file);
      H5Pclose(property_list_file);
    }

    if (opacity_size != temp_size) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
          << "Opacity array size must match coordinate array size." << std::endl
          << "opacity_size = " << opacity_size << ", temp_size = " << temp_size << std::endl;
      ATHENA_ERROR(msg);
    }

    opacity_1d.NewAthenaArray(opacity_size);
    int count_opacity[1] = {opacity_size};

    HDF5ReadRealArray(fn.c_str(), var_names[ivar], 1, start_file, count_opacity,
                      1, start_mem, count_opacity, opacity_1d);

    // Store 1D opacity data in 2D structure (j=0, i=index)
    // Apply unit conversion from physical units (cm^2/g) to code units
    for (int i = 0; i < nx1; ++i) {
      puser_table->data(ivar, 0, i) = opacity_1d(i) / puser_table->opacity_unit;
    }

    opacity_1d.DeleteAthenaArray();
  }

  // Clean up coordinate arrays
  temp_array.DeleteAthenaArray();
  density_array.DeleteAthenaArray();

  // Initialize tables (if needed for unit conversion)
  bool read_tables = pin->GetOrAddBoolean("mgfld", "opacity_table_read_tables", false);
  if (!read_tables) {
    puser_table->OpacityTables.NewAthenaArray(puser_table->nVar);
    for (int i=0; i<puser_table->nVar; ++i) puser_table->OpacityTables(i) = 1.0;
  }
  #else
  std::stringstream msg;
  msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
      << "HDF5 support not enabled. Reconfigure with -hdf5." << std::endl;
  ATHENA_ERROR(msg);
  #endif
}

//----------------------------------------------------------------------------------------
//! \fn UserOpacityTable::UserOpacityTable(ParameterInput *pin)
//  \brief Constructor for UserOpacityTable class, reads the opacity table from file.
//   \param pin Pointer to ParameterInput object containing user-defined parameters.
//   \note The table is read from a file specified in the parameter input, and the size of
//         the table is set based on the data read.
UserOpacityTable::UserOpacityTable(ParameterInput *pin) : InterpTable2D() {
  std::string opacity_fn, opacity_file_type;

  // Get file name and type from parameters
  opacity_fn = pin->GetString("mgfld", "opacity_table_file");
  opacity_file_type = pin->GetOrAddString("mgfld", "opacity_table_file_type", "auto");

  // Get units
  // tempUnit = pin->GetReal("mgfld", "opacity_temp_unit");
  // densityUnit = pin->GetReal("mgfld", "opacity_density_unit");

  rho_unit = pin->GetReal("hydro", "rho_unit");
  Real egas_unit = pin->GetReal("hydro", "egas_unit");
  Real time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  Real leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "time_unit or leng_unit must be specified in block 'hydro'.";
    ATHENA_ERROR(msg);
  } else if (time_unit > 0.0 && leng_unit > 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "time_unit and leng_unit cannot be specified at the same time.";
    ATHENA_ERROR(msg);
  }
  Real pres_unit = egas_unit;
  // Rgas in cgs
  Real Rgas = 8.31451e+7; // erg/(mol*K)
  Real mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;

  Real vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;
  opacity_unit = 1.0/(rho_unit*leng_unit); // cm^2/g

  // Auto-detect file type based on extension if not specified
  if (opacity_file_type.compare("auto") == 0) {
    std::string ext = opacity_fn.substr(opacity_fn.find_last_of(".") + 1);
    if (ext.compare("hdf5")*ext.compare("h5") == 0) {
      opacity_file_type.assign("hdf5");
    } else if (ext.compare("tab")*ext.compare("txt")*ext.compare("ascii") == 0) {
      opacity_file_type.assign("ascii");
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
          << "Cannot auto-detect file type for '" << opacity_fn << "'." << std::endl
          << "Please specify opacity_table_file_type explicitly." << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // Read the table based on file type
  if (opacity_file_type.compare("hdf5") == 0) { // HDF5 table
    #ifdef HDF5OUTPUT
    ReadHDF5OpacityTable(opacity_fn, this, pin);
    #else
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
        << "HDF5 support not enabled. Reconfigure with -hdf5." << std::endl;
    ATHENA_ERROR(msg);
    #endif
  } else if (opacity_file_type.compare("ascii") == 0) { // ASCII/text table
    ReadAsciiOpacityTable(opacity_fn, this, pin);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
        << "Opacity table of type '" << opacity_file_type << "' not recognized."  << std::endl
        << "Options are 'ascii' and 'hdf5'." << std::endl;
    ATHENA_ERROR(msg);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real UserOpacityTable::GetOpacity(int var_index, Real x2, Real x1)
//  \brief Gets interpolated opacity data from the table using generic coordinates.
//   \param var_index Index of the opacity variable
//   \param x2 Second coordinate (typically density or log density)
//   \param x1 First coordinate (typically temperature or log temperature)
//   \return Interpolated opacity value in code units
//   \note Opacity data is stored in code units after unit conversion during loading
Real UserOpacityTable::GetOpacity(int var_index, Real x2, Real x1) {
  // For 1D paired data (nDensity = 1), use nearest neighbor approach
  if (nDensity == 1 && temp_coords.GetDim1() > 0) {
    // Find the nearest point based on both temperature and density
    Real min_distance = 1e30;
    int best_idx = 0;

    for (int i = 0; i < nTemp; ++i) {
      // Calculate distance in log space (since both T and rho span many orders of magnitude)
      Real temp_distance = std::log10(x1) - std::log10(temp_coords(i));
      Real density_distance = std::log10(x2) - std::log10(density_coords(i));
      Real distance = std::sqrt(temp_distance*temp_distance + density_distance*density_distance);
      std::cout << "Distance for index " << i << ": " << distance << std::endl; // Debug output

      // Update minimum distance and best index
      if (distance < min_distance) {
        min_distance = distance;
        best_idx = i;
      }
    }

    return data(var_index, 0, best_idx);
  } else {
    // For 2D data, use the normal interpolation
    return interpolate(var_index, x2, x1);
  }
}

//----------------------------------------------------------------------------------------
// UserOpacityTable destructor
UserOpacityTable::~UserOpacityTable() {
  // AthenaArray destructor will handle cleanup automatically
  if (OpacityTables.GetDim1() > 0) {
    OpacityTables.DeleteAthenaArray();
  }
  // Note: temp_coords and density_coords are AthenaArray objects,
  // their destructors will handle cleanup automatically
}
