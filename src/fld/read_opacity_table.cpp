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
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../inputs/ascii_table_reader.hpp"
#include "../inputs/hdf5_reader.hpp"
#include "../parameter_input.hpp"
#include "../utils/interp_table.hpp"
#include "fld.hpp"
#include "opacity_table.hpp"

#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

// Order of datafields for HDF5 opacity tables
// These variable names should match the dataset names in your HDF5 opacity file
// Order must match enum OpacityIndex {SIGMA_P=0, SIGMA_R=1} in rad_fld.hpp
const char *opacity_var_names[] = {"planck_mean_opacity", "rosseland_mean_opacity"};

#ifdef HDF5OUTPUT
namespace {
constexpr const char *kOpacityBlockPrimary = "fld";

bool OpacityParamExists(ParameterInput *pin, const std::string &name) {
  return (pin->DoesParameterExist(kOpacityBlockPrimary, name) != 0);
}

std::string GetOpacityStringOrDefault(ParameterInput *pin, const std::string &name,
                                      const std::string &default_value) {
  if (pin->DoesParameterExist(kOpacityBlockPrimary, name)) {
    return pin->GetString(kOpacityBlockPrimary, name);
  }
  return default_value;
}

bool GetOpacityBoolOrDefault(ParameterInput *pin, const std::string &name,
                             bool default_value) {
  if (pin->DoesParameterExist(kOpacityBlockPrimary, name)) {
    return pin->GetBoolean(kOpacityBlockPrimary, name);
  }
  return default_value;
}

bool FindExistingDataset(hid_t file, const std::vector<std::string> &candidates,
                         std::string *found_path) {
  for (const auto &path : candidates) {
    htri_t exists = H5Lexists(file, path.c_str(), H5P_DEFAULT);
    if (exists > 0) {
      *found_path = path;
      return true;
    }
  }
  return false;
}

bool PathLooksLikeLog10Opacity(const std::string &path) {
  return (path.find("log10_planck") != std::string::npos ||
          path.find("log10_rosseland") != std::string::npos ||
          path.find("log_planck") != std::string::npos ||
          path.find("log_rosseland") != std::string::npos);
}

std::string ResolveDatasetPath(hid_t file, const std::string &configured_name,
                               const std::vector<std::string> &fallback_candidates,
                               const std::string &dataset_purpose) {
  if (configured_name != "auto") {
    if (H5Lexists(file, configured_name.c_str(), H5P_DEFAULT) > 0) {
      return configured_name;
    }
    if (!configured_name.empty() && configured_name[0] != '/') {
      std::string with_slash = "/" + configured_name;
      if (H5Lexists(file, with_slash.c_str(), H5P_DEFAULT) > 0) {
        return with_slash;
      }
    }
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Configured dataset for " << dataset_purpose << " ('"
        << configured_name << "') was not found in the HDF5 file." << std::endl;
    ATHENA_ERROR(msg);
  }

  std::string detected_path;
  if (FindExistingDataset(file, fallback_candidates, &detected_path)) {
    return detected_path;
  }

  std::stringstream msg;
  msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
      << "Could not auto-detect dataset for " << dataset_purpose << "." << std::endl;
  ATHENA_ERROR(msg);
  return "";
}

int Read1DDatasetSize(const std::string &fn, const std::string &dataset_path) {
  hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
  hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
  hid_t dataset = H5Dopen(file, dataset_path.c_str(), H5P_DEFAULT);
  hid_t dspace = H5Dget_space(dataset);
  int ndims = H5Sget_simple_extent_ndims(dspace);
  if (ndims != 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Dataset '" << dataset_path << "' must be 1D, found " << ndims << "D." << std::endl;
    ATHENA_ERROR(msg);
  }
  hsize_t dims[1];
  H5Sget_simple_extent_dims(dspace, dims, NULL);
  H5Sclose(dspace);
  H5Dclose(dataset);
  H5Fclose(file);
  H5Pclose(property_list_file);
  return static_cast<int>(dims[0]);
}

void Read2DDatasetShape(const std::string &fn, const std::string &dataset_path,
                        hsize_t dims_out[2]) {
  hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
  hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
  hid_t dataset = H5Dopen(file, dataset_path.c_str(), H5P_DEFAULT);
  hid_t dspace = H5Dget_space(dataset);
  int ndims = H5Sget_simple_extent_ndims(dspace);
  if (ndims != 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Opacity dataset '" << dataset_path << "' must be 2D, found "
        << ndims << "D." << std::endl;
    ATHENA_ERROR(msg);
  }
  H5Sget_simple_extent_dims(dspace, dims_out, NULL);
  H5Sclose(dspace);
  H5Dclose(dataset);
  H5Fclose(file);
  H5Pclose(property_list_file);
}
}  // namespace
#endif

//----------------------------------------------------------------------------------------
//! \fn void ReadAsciiOpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin)
//  \brief Read data from ascii opacity table and initialize interpolated table.
void ReadAsciiOpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin) {
  AthenaArray<Real> *ptables = nullptr;
  if (puser_table->use_tables) ptables = &puser_table->OpacityTables;

  // If use_tables then OpacityTables.NewAthenaArray is called in ASCIITableLoader
  ASCIITableLoader(fn.c_str(), *puser_table, ptables);
  puser_table->GetSize(puser_table->nVar, puser_table->nPressure, puser_table->nTemp);
  puser_table->GetX2lim(puser_table->pressureMin, puser_table->pressureMax);
  puser_table->GetX1lim(puser_table->tempMin, puser_table->tempMax);

  if (!puser_table->use_tables) {
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
  int nvar = RadFLD2::NOPACITY;
  std::string temp_path, x2_path;
  std::string var_paths[RadFLD2::NOPACITY];
  bool values_are_log10 = false;
  UserOpacityTable::X2AxisKind x2_axis_kind = puser_table->x2_axis_kind;

  std::vector<std::string> x2_candidates;
  if (x2_axis_kind == UserOpacityTable::X2AxisKind::pressure) {
    x2_candidates = {"log_pressure", "/log_pressure",
                     "axes/log10_pressure", "/axes/log10_pressure",
                     "log10_pressure", "/log10_pressure"};
  } else if (x2_axis_kind == UserOpacityTable::X2AxisKind::density) {
    x2_candidates = {"log_density", "/log_density",
                     "log10_density", "/log10_density",
                     "axes/log10_density", "/axes/log10_density"};
  } else {
    x2_candidates = {"logR", "/logR",
                     "log_r", "/log_r",
                     "axes/logR", "/axes/logR",
                     "axes/log_r", "/axes/log_r"};
  }

  // Resolve dataset paths. Axis kind is controlled by input parameter:
  // nrfld/opacity_table_axis (or mgfld/opacity_table_axis) = tp, trho, tr
  {
    hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);

    temp_path = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_temperature_dataset", "auto"),
        {"log_temperature", "/log_temperature",
         "axes/log10_temperature", "/axes/log10_temperature",
         "log10_temperature", "/log10_temperature"},
        "temperature axis");

    x2_path = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_x2_dataset", "auto"),
        x2_candidates,
        "x2 axis");

    var_paths[RadFLD2::SIGMA_P] = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_planck_dataset", "auto"),
        {"kappa/log10_planck", "/kappa/log10_planck",
         "log10_planck", "/log10_planck",
         opacity_var_names[RadFLD2::SIGMA_P],
         "/planck_mean_opacity",
         "kappa/planck", "/kappa/planck", "planck", "/planck"},
        "Planck opacity");

    var_paths[RadFLD2::SIGMA_R] = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_rosseland_dataset", "auto"),
        {"kappa/log10_rosseland", "/kappa/log10_rosseland",
         "log10_rosseland", "/log10_rosseland",
         opacity_var_names[RadFLD2::SIGMA_R],
         "/rosseland_mean_opacity",
         "kappa/rosseland", "/kappa/rosseland", "rosseland", "/rosseland"},
        "Rosseland opacity");

    values_are_log10 = PathLooksLikeLog10Opacity(var_paths[RadFLD2::SIGMA_P]) &&
                       PathLooksLikeLog10Opacity(var_paths[RadFLD2::SIGMA_R]);

    H5Fclose(file);
    H5Pclose(property_list_file);
  }

  // Read 2D grid format: separate 1D coordinate arrays and 2D opacity grids
  AthenaArray<Real> temp_array, x2_array;

  int temp_size = Read1DDatasetSize(fn, temp_path);
  int pressure_size = Read1DDatasetSize(fn, x2_path);

  // Set up proper 2D table structure
  puser_table->SetSize(nvar, pressure_size, temp_size);  // nvar, nx2=pressure_size, nx1=temp_size
  puser_table->nVar = nvar;
  puser_table->nTemp = temp_size;
  puser_table->nPressure = pressure_size;
  puser_table->x2_axis_kind = x2_axis_kind;
  puser_table->values_are_log10 = values_are_log10;

  // Read coordinate arrays
  temp_array.NewAthenaArray(temp_size);
  x2_array.NewAthenaArray(pressure_size);

  int start_file[1] = {0};
  int start_mem[1] = {0};
  int count_temp[1] = {temp_size};
  int count_pressure[1] = {pressure_size};

  HDF5ReadRealArray(fn.c_str(), temp_path.c_str(), 1, start_file, count_temp,
                    1, start_mem, count_temp, temp_array);
  HDF5ReadRealArray(fn.c_str(), x2_path.c_str(), 1, start_file, count_pressure,
                    1, start_mem, count_pressure, x2_array);

  // Set coordinate limits in physical units
  puser_table->tempMin = temp_array(0);
  puser_table->tempMax = temp_array(temp_size - 1);
  puser_table->pressureMin = x2_array(0);
  puser_table->pressureMax = x2_array(pressure_size - 1);

  // Set grid bounds in physical units for InterpTable2D
  puser_table->SetX1lim(puser_table->tempMin, puser_table->tempMax);
  puser_table->SetX2lim(puser_table->pressureMin, puser_table->pressureMax);

  // Read 2D opacity data arrays
  for (int ivar = 0; ivar < nvar; ++ivar) {
    hsize_t opacity_dims[2];
    Read2DDatasetShape(fn, var_paths[ivar], opacity_dims);

    // Verify dimensions match coordinate arrays
    if (static_cast<int>(opacity_dims[0]) != pressure_size ||
        static_cast<int>(opacity_dims[1]) != temp_size) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
          << "Opacity array dimensions [" << opacity_dims[0] << "," << opacity_dims[1]
          << "] must match coordinate arrays [" << pressure_size << "," << temp_size << "]" << std::endl;
      ATHENA_ERROR(msg);
    }

    // Read 2D opacity data
    AthenaArray<Real> opacity_2d;
    opacity_2d.NewAthenaArray(pressure_size, temp_size);

    int start_file_2d[2] = {0, 0};
    int start_mem_2d[2] = {0, 0};
    int count_2d[2] = {pressure_size, temp_size};

    HDF5ReadRealArray(fn.c_str(), var_paths[ivar].c_str(), 2, start_file_2d, count_2d,
                      2, start_mem_2d, count_2d, opacity_2d);

    // Store 2D opacity data. HDF5 data is opacity[pressure_idx, temp_idx],
    // InterpTable2D expects data(ivar, j, i).
    for (int j = 0; j < pressure_size; ++j) {
      for (int i = 0; i < temp_size; ++i) {
        puser_table->data(ivar, j, i) = opacity_2d(j, i);
      }
    }

    opacity_2d.DeleteAthenaArray();
  }

  // Clean up coordinate arrays
  temp_array.DeleteAthenaArray();
  x2_array.DeleteAthenaArray();

  if (!puser_table->use_tables) {
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
  mean_molecular_weight = pin->GetOrAddReal("hydro", "mu", 1.0);
  std::string axis_mode = GetOpacityStringOrDefault(pin, "opacity_table_axis", "tp");
  if (axis_mode.compare("tp") == 0 || axis_mode.compare("t-p") == 0) {
    x2_axis_kind = X2AxisKind::pressure;
  } else if (axis_mode.compare("trho") == 0 || axis_mode.compare("t-rho") == 0) {
    x2_axis_kind = X2AxisKind::density;
  } else if (axis_mode.compare("tr") == 0 || axis_mode.compare("t-r") == 0) {
    x2_axis_kind = X2AxisKind::opal_r;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
        << "Unknown opacity_table_axis='" << axis_mode << "' in <nrfld>/<mgfld>." << std::endl
        << "Options: tp, trho, tr." << std::endl;
    ATHENA_ERROR(msg);
  }

  use_tables = GetOpacityBoolOrDefault(pin, "use_opacity_table", false);
  if (!use_tables) return;
  std::string opacity_fn, opacity_file_type;

  // Get file name and type from parameters
  if (!OpacityParamExists(pin, "opacity_table_file")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
        << "Opacity table is enabled, but 'opacity_table_file' was not found in either "
        << "<nrfld> or <mgfld> block." << std::endl;
    ATHENA_ERROR(msg);
  }
  opacity_fn = GetOpacityStringOrDefault(pin, "opacity_table_file", "");
  opacity_file_type = GetOpacityStringOrDefault(pin, "opacity_table_file_type", "auto");

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
//! \fn Real UserOpacityTable::GetOpacity(int var_index, Real density, Real temperature)
//  \brief Gets interpolated opacity data from the 2D table using bilinear interpolation.
//   \param var_index Index of the opacity variable
//   \param density Density in physical units (g/cm^3)
//   \param temperature Temperature in physical units (K)
//   \note The interpolation is done on a logarithmic scale for both axes.
Real UserOpacityTable::GetOpacity(int var_index, Real density, Real temperature) {
  return GetOpacityFromRhoT(var_index, density, temperature);
}

Real UserOpacityTable::GetOpacityFromPT(int var_index, Real pressure, Real temperature) {
  if (!use_tables) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromPT" << std::endl
        << "Opacity tables are not enabled. Set 'use_opacity_table=true' in "
        << "<fld> block." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (pressure <= 0.0 || temperature <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromPT" << std::endl
        << "Pressure and temperature must be positive. Received P=" << pressure
        << ", T=" << temperature << std::endl;
    ATHENA_ERROR(msg);
  }

  constexpr Real r_gas_cgs = 8.314462618e7;
  Real density = pressure*mean_molecular_weight/(r_gas_cgs*temperature);
  return GetOpacityFromRhoT(var_index, density, temperature);
}

Real UserOpacityTable::GetOpacityFromRhoT(int var_index, Real density, Real temperature) {
  if (!use_tables) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << "Opacity tables are not enabled. Set 'use_opacity_table=true' in "
        << "<fld> block." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (density <= 0.0 || temperature <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << "Density and temperature must be positive. Received rho=" << density
        << ", T=" << temperature << std::endl;
    ATHENA_ERROR(msg);
  }

  constexpr Real r_gas_cgs = 8.314462618e7;
  Real log_temperature = std::log10(temperature);
  Real log_x2 = 0.0;
  if (x2_axis_kind == X2AxisKind::density) {
    log_x2 = std::log10(density);
  } else if (x2_axis_kind == X2AxisKind::pressure) {
    Real pressure = density*r_gas_cgs*temperature/mean_molecular_weight;
    log_x2 = std::log10(pressure);
  } else {
    log_x2 = std::log10(density) - 3.0*log_temperature + 18.0;
  }
  Real result = interpolate(var_index, log_x2, log_temperature);
  if (values_are_log10) {
    result = std::pow(static_cast<Real>(10.0), result);
  }
  if (result < 0.0) {
    result = TINY_NUMBER;
    std::cerr << "Warning: Negative opacity value encountered. Returning TINY_NUMBER instead."
              << std::endl;
    std::cerr << "log_x2: " << log_x2 << ", log_temperature: " << log_temperature
              << std::endl;
    std::cerr << "Interpolated result: " << result << std::endl;
  }
  return result;
}

//----------------------------------------------------------------------------------------
// UserOpacityTable destructor
UserOpacityTable::~UserOpacityTable() {
  // AthenaArray destructor will handle cleanup automatically
  if (OpacityTables.GetDim1() > 0) {
    OpacityTables.DeleteAthenaArray();
  }
  // Note: temp_coords and pressure_coords are AthenaArray objects,
  // their destructors will handle cleanup automatically
}
