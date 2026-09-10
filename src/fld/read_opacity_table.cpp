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
#include <algorithm>
#include <cmath>   // sqrt()
#include <cstddef>
#include <fstream>
#include <iostream> // ifstream
#include <limits>
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../inputs/ascii_table_reader.hpp"
#include "../inputs/hdf5_reader.hpp"
#include "../parameter_input.hpp"
#include "../utils/interp_table.hpp"
#include "fld.hpp"
#include "opacity_table.hpp"

#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Order of datafields for HDF5 opacity tables
// These variable names should match the dataset names in your HDF5 opacity file
// Order must match RadFLD::OpacityIndex in fld.hpp.
const char *opacity_var_names[] = {"planck_mean_opacity", "rosseland_mean_opacity"};

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

UserOpacityTable::DomainPolicy ParseDomainPolicy(ParameterInput *pin) {
  const std::string policy =
      GetOpacityStringOrDefault(pin, "opacity_table_domain_policy", "extrapolate");
  if (policy == "error") return UserOpacityTable::DomainPolicy::error;
  if (policy == "clamp") return UserOpacityTable::DomainPolicy::clamp;
  if (policy == "extrapolate") return UserOpacityTable::DomainPolicy::extrapolate;

  std::stringstream msg;
  msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
      << "fld/opacity_table_domain_policy must be 'error', 'clamp', or "
      << "'extrapolate', got '" << policy << "'." << std::endl;
  ATHENA_ERROR(msg);
  return UserOpacityTable::DomainPolicy::extrapolate;
}

bool ParseExplicitOpacityFormat(ParameterInput *pin, const std::string &parameter,
                                bool *is_auto) {
  std::string format = GetOpacityStringOrDefault(pin, parameter, "auto");
  if (format == "auto") {
    *is_auto = true;
    return false;
  }
  *is_auto = false;
  if (format == "linear") return false;
  if (format == "log10") return true;

  std::stringstream msg;
  msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
      << "fld/" << parameter << " must be 'auto', 'linear', or 'log10', got '"
      << format << "'." << std::endl;
  ATHENA_ERROR(msg);
  return false;
}

void SetAsciiOpacityFormats(ParameterInput *pin, UserOpacityTable *table) {
  bool is_auto = false;
  table->values_are_log10[RadFLD::SIGMA_P] =
      ParseExplicitOpacityFormat(pin, "opacity_table_planck_format", &is_auto);
  // The historical ASCII opacity format stores linear values and has no
  // metadata from which to infer another representation.
  if (is_auto) table->values_are_log10[RadFLD::SIGMA_P] = false;

  table->values_are_log10[RadFLD::SIGMA_R] =
      ParseExplicitOpacityFormat(pin, "opacity_table_rosseland_format", &is_auto);
  if (is_auto) table->values_are_log10[RadFLD::SIGMA_R] = false;
}

void ValidateAsciiOpacitySchema(const std::string &filename, UserOpacityTable *table) {
  if (table->nVar != RadFLD::NOPACITY) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadAsciiOpacityTable" << std::endl
        << "Opacity table shape must contain exactly " << RadFLD::NOPACITY
        << " fields (Planck, Rosseland), found " << table->nVar << " in '"
        << filename << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (table->nTemp < 2 || table->nPressure < 2 ||
      !std::isfinite(table->tempMin) || !std::isfinite(table->tempMax) ||
      !std::isfinite(table->pressureMin) || !std::isfinite(table->pressureMax) ||
      table->tempMin >= table->tempMax || table->pressureMin >= table->pressureMax) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadAsciiOpacityTable" << std::endl
        << "Opacity axes must have at least two points and finite, increasing limits in '"
        << filename << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
}

const char *OpacityFieldName(int field) {
  return field == RadFLD::SIGMA_P ? "Planck" : "Rosseland";
}

void ValidateOpacityFields(const std::string &filename, UserOpacityTable *table) {
  for (int field = 0; field < table->nVar; ++field) {
    for (int j = 0; j < table->nPressure; ++j) {
      for (int i = 0; i < table->nTemp; ++i) {
        const Real stored = table->data(field, j, i);
        Real decoded = stored;
        if (table->values_are_log10[field] && std::isfinite(stored)) {
          decoded = std::pow(static_cast<Real>(10.0), stored);
        }
        if (!std::isfinite(stored) || !std::isfinite(decoded) || decoded <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in UserOpacityTable table validation" << std::endl
              << OpacityFieldName(field) << " opacity field in '" << filename
              << "' contains an invalid value at [x2=" << j << ", temperature=" << i
              << "]: stored=" << stored;
          if (table->values_are_log10[field]) msg << ", decoded=" << decoded;
          msg << ". Every stored value must be finite and every decoded opacity "
              << "must be finite and positive." << std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }
  }
}

#ifdef HDF5OUTPUT
class HDF5Handle {
 public:
  using CloseFunction = herr_t (*)(hid_t);

  HDF5Handle(hid_t id, CloseFunction close) : id_(id), close_(close) {}
  ~HDF5Handle() {
    if (id_ >= 0) close_(id_);
  }
  HDF5Handle(const HDF5Handle &) = delete;
  HDF5Handle &operator=(const HDF5Handle &) = delete;
  operator hid_t() const { return id_; }
  bool valid() const { return id_ >= 0; }

 private:
  hid_t id_;
  CloseFunction close_;
};

void CheckHDF5Link(htri_t status, const std::string &filename,
                   const std::string &dataset_path) {
  if (status < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Failed while checking HDF5 dataset '" << dataset_path
        << "' in file '" << filename << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
}

bool FindExistingDataset(hid_t file, const std::vector<std::string> &candidates,
                         const std::string &filename, std::string *found_path) {
  for (const auto &path : candidates) {
    htri_t exists = H5Lexists(file, path.c_str(), H5P_DEFAULT);
    CheckHDF5Link(exists, filename, path);
    if (exists > 0) {
      *found_path = path;
      return true;
    }
  }
  return false;
}

std::string DatasetBasename(const std::string &path) {
  std::string::size_type separator = path.find_last_of('/');
  return separator == std::string::npos ? path : path.substr(separator + 1);
}

bool InferHDF5OpacityFormat(ParameterInput *pin, const std::string &parameter,
                           const std::string &path, int opacity_index) {
  bool is_auto = false;
  bool values_are_log10 = ParseExplicitOpacityFormat(pin, parameter, &is_auto);
  if (!is_auto) return values_are_log10;

  const std::string name = DatasetBasename(path);
  const bool is_planck = (opacity_index == RadFLD::SIGMA_P);
  const std::string field = is_planck ? "planck" : "rosseland";
  if (name == "log10_" + field || name == "log_" + field) return true;
  if (name == field || name == field + "_mean_opacity") return false;

  std::stringstream msg;
  msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
      << "Cannot infer whether " << field << " dataset '" << path
      << "' stores linear or log10 values. Set fld/" << parameter
      << " explicitly to 'linear' or 'log10'." << std::endl;
  ATHENA_ERROR(msg);
  return false;
}

void ValidateUniformAxis(const AthenaArray<Real> &axis, int size,
                         const std::string &dataset_path, Real storage_epsilon) {
  if (size < 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Axis dataset '" << dataset_path << "' must contain at least two points, found "
        << size << "." << std::endl;
    ATHENA_ERROR(msg);
  }

  for (int i = 0; i < size; ++i) {
    if (!std::isfinite(axis(i))) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
          << "Axis dataset '" << dataset_path << "' contains a non-finite value at index "
          << i << "." << std::endl;
      ATHENA_ERROR(msg);
    }
    if (i > 0 && axis(i) <= axis(i - 1)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
          << "Axis dataset '" << dataset_path
          << "' must be strictly increasing; values at indices " << i - 1 << " and " << i
          << " are " << axis(i - 1) << " and " << axis(i) << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  const Real spacing = (axis(size - 1) - axis(0))/static_cast<Real>(size - 1);
  Real scale = std::abs(axis(0));
  if (std::abs(axis(size - 1)) > scale) scale = std::abs(axis(size - 1));
  if (std::abs(spacing) > scale) scale = std::abs(spacing);
  const Real tolerance = 64.0*storage_epsilon*scale;
  for (int i = 1; i < size - 1; ++i) {
    const Real expected = axis(0) + static_cast<Real>(i)*spacing;
    if (std::abs(axis(i) - expected) > tolerance) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
          << "Axis dataset '" << dataset_path
          << "' is not uniformly spaced; index " << i << " has value " << axis(i)
          << ", expected " << expected << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
}

std::string ResolveDatasetPath(hid_t file, const std::string &configured_name,
                               const std::vector<std::string> &fallback_candidates,
                               const std::string &dataset_purpose,
                               const std::string &filename) {
  if (configured_name != "auto") {
    htri_t exists = H5Lexists(file, configured_name.c_str(), H5P_DEFAULT);
    CheckHDF5Link(exists, filename, configured_name);
    if (exists > 0) {
      return configured_name;
    }
    if (!configured_name.empty() && configured_name[0] != '/') {
      std::string with_slash = "/" + configured_name;
      exists = H5Lexists(file, with_slash.c_str(), H5P_DEFAULT);
      CheckHDF5Link(exists, filename, with_slash);
      if (exists > 0) {
        return with_slash;
      }
    }
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Configured dataset for " << dataset_purpose << " ('"
        << configured_name << "') was not found in HDF5 file '" << filename
        << "'." << std::endl;
    ATHENA_ERROR(msg);
  }

  std::string detected_path;
  if (FindExistingDataset(file, fallback_candidates, filename, &detected_path)) {
    return detected_path;
  }

  std::stringstream msg;
  msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
      << "Could not auto-detect dataset for " << dataset_purpose
      << " in HDF5 file '" << filename << "'." << std::endl;
  ATHENA_ERROR(msg);
  return "";
}

int Read1DDatasetSize(const std::string &fn, const std::string &dataset_path,
                      Real *storage_epsilon) {
  HDF5Handle property_list_file(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
  if (!property_list_file.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not create an HDF5 file-access property list for '" << fn << "'."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle file(H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file), H5Fclose);
  if (!file.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not open HDF5 file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle dataset(H5Dopen(file, dataset_path.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dataset.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not open HDF5 dataset '" << dataset_path << "' in file '"
        << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle dspace(H5Dget_space(dataset), H5Sclose);
  if (!dspace.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not get the dataspace for HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  int ndims = H5Sget_simple_extent_ndims(dspace);
  if (ndims < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not read the rank of HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (ndims != 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Dataset '" << dataset_path << "' must be 1D, found " << ndims << "D." << std::endl;
    ATHENA_ERROR(msg);
  }
  hsize_t dims[1];
  if (H5Sget_simple_extent_dims(dspace, dims, NULL) < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not read the shape of HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle datatype(H5Dget_type(dataset), H5Tclose);
  if (!datatype.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not get the datatype for HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  const H5T_class_t datatype_class = H5Tget_class(datatype);
  const std::size_t datatype_size = H5Tget_size(datatype);
  if (datatype_class == H5T_NO_CLASS || datatype_size == 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not inspect the datatype of HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  *storage_epsilon = std::numeric_limits<Real>::epsilon();
  if (datatype_class == H5T_FLOAT && datatype_size <= sizeof(float) &&
      std::numeric_limits<float>::epsilon() > *storage_epsilon) {
    *storage_epsilon = std::numeric_limits<float>::epsilon();
  }
  return static_cast<int>(dims[0]);
}

void Read2DDatasetShape(const std::string &fn, const std::string &dataset_path,
                        hsize_t dims_out[2]) {
  HDF5Handle property_list_file(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
  if (!property_list_file.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not create an HDF5 file-access property list for '" << fn << "'."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle file(H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file), H5Fclose);
  if (!file.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not open HDF5 file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle dataset(H5Dopen(file, dataset_path.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dataset.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not open HDF5 dataset '" << dataset_path << "' in file '"
        << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle dspace(H5Dget_space(dataset), H5Sclose);
  if (!dspace.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not get the dataspace for HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  int ndims = H5Sget_simple_extent_ndims(dspace);
  if (ndims < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not read the rank of HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (ndims != 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Opacity dataset '" << dataset_path << "' must be 2D, found "
        << ndims << "D." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (H5Sget_simple_extent_dims(dspace, dims_out, NULL) < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Could not read the shape of HDF5 dataset '" << dataset_path
        << "' in file '" << fn << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
}
#endif
}  // namespace

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
  ValidateAsciiOpacitySchema(fn, puser_table);
  SetAsciiOpacityFormats(pin, puser_table);
  ValidateOpacityFields(fn, puser_table);

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
  int nvar = RadFLD::NOPACITY;
  std::string temp_path, x2_path;
  std::string var_paths[RadFLD::NOPACITY];
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
  // fld/opacity_table_axis = tp, trho, tr
  {
    HDF5Handle property_list_file(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    if (!property_list_file.valid()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
          << "Could not create an HDF5 file-access property list for '" << fn << "'."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    HDF5Handle file(H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file), H5Fclose);
    if (!file.valid()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
          << "Could not open HDF5 file '" << fn << "'." << std::endl;
      ATHENA_ERROR(msg);
    }

    temp_path = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_temperature_dataset", "auto"),
        {"log_temperature", "/log_temperature",
         "axes/log10_temperature", "/axes/log10_temperature",
         "log10_temperature", "/log10_temperature"},
        "temperature axis", fn);

    x2_path = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_x2_dataset", "auto"),
        x2_candidates,
        "x2 axis", fn);

    var_paths[RadFLD::SIGMA_P] = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_planck_dataset", "auto"),
        {"kappa/log10_planck", "/kappa/log10_planck",
         "log10_planck", "/log10_planck",
         opacity_var_names[RadFLD::SIGMA_P],
         "/planck_mean_opacity",
         "kappa/planck", "/kappa/planck", "planck", "/planck"},
        "Planck opacity", fn);

    var_paths[RadFLD::SIGMA_R] = ResolveDatasetPath(
        file,
        GetOpacityStringOrDefault(pin, "opacity_table_rosseland_dataset", "auto"),
        {"kappa/log10_rosseland", "/kappa/log10_rosseland",
         "log10_rosseland", "/log10_rosseland",
         opacity_var_names[RadFLD::SIGMA_R],
         "/rosseland_mean_opacity",
         "kappa/rosseland", "/kappa/rosseland", "rosseland", "/rosseland"},
        "Rosseland opacity", fn);

    puser_table->values_are_log10[RadFLD::SIGMA_P] = InferHDF5OpacityFormat(
        pin, "opacity_table_planck_format", var_paths[RadFLD::SIGMA_P], RadFLD::SIGMA_P);
    puser_table->values_are_log10[RadFLD::SIGMA_R] = InferHDF5OpacityFormat(
        pin, "opacity_table_rosseland_format", var_paths[RadFLD::SIGMA_R], RadFLD::SIGMA_R);

  }

  // Read 2D grid format: separate 1D coordinate arrays and 2D opacity grids
  AthenaArray<Real> temp_array, x2_array;

  Real temp_storage_epsilon, x2_storage_epsilon;
  int temp_size = Read1DDatasetSize(fn, temp_path, &temp_storage_epsilon);
  int pressure_size = Read1DDatasetSize(fn, x2_path, &x2_storage_epsilon);

  if (temp_size < 2 || pressure_size < 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
        << "Opacity axes must each contain at least two points; '" << temp_path << "' has "
        << temp_size << " and '" << x2_path << "' has " << pressure_size << "." << std::endl;
    ATHENA_ERROR(msg);
  }

  // Set up proper 2D table structure
  puser_table->SetSize(nvar, pressure_size, temp_size);  // nvar, nx2=pressure_size, nx1=temp_size
  puser_table->nVar = nvar;
  puser_table->nTemp = temp_size;
  puser_table->nPressure = pressure_size;
  puser_table->x2_axis_kind = x2_axis_kind;

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

  ValidateUniformAxis(temp_array, temp_size, temp_path, temp_storage_epsilon);
  ValidateUniformAxis(x2_array, pressure_size, x2_path, x2_storage_epsilon);

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

  ValidateOpacityFields(fn, puser_table);

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
        << "Unknown opacity_table_axis='" << axis_mode << "' in <fld>." << std::endl
        << "Options: tp, trho, tr." << std::endl;
    ATHENA_ERROR(msg);
  }

  use_tables = GetOpacityBoolOrDefault(pin, "use_opacity_table", false);
  if (!use_tables) return;
  domain_policy = ParseDomainPolicy(pin);
  if (!std::isfinite(mean_molecular_weight) || mean_molecular_weight <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
        << "hydro/mu must be finite and positive because FLD opacity TP<->rhoT "
        << "conversion uses a fixed-mu ideal-gas relation. Received mu="
        << mean_molecular_weight << "." << std::endl;
    ATHENA_ERROR(msg);
  }
  std::string opacity_fn, opacity_file_type;

  // Get file name and type from parameters
  if (!OpacityParamExists(pin, "opacity_table_file")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
        << "Opacity table is enabled, but 'opacity_table_file' was not found in "
        << "the <fld> block." << std::endl;
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

  if (!std::isfinite(pressure) || !std::isfinite(temperature) ||
      pressure <= 0.0 || temperature <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromPT" << std::endl
        << "Pressure and temperature must be finite and positive. Received P=" << pressure
        << ", T=" << temperature << std::endl;
    ATHENA_ERROR(msg);
  }

  constexpr Real r_gas_cgs = 8.314462618e7;
  Real density = pressure*mean_molecular_weight/(r_gas_cgs*temperature);
  if (!std::isfinite(density) || density <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromPT" << std::endl
        << "Fixed-mu ideal-gas conversion produced invalid density rho=" << density
        << " from P=" << pressure << ", T=" << temperature
        << ", mu=" << mean_molecular_weight << "." << std::endl;
    ATHENA_ERROR(msg);
  }
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

  if (var_index < 0 || var_index >= nVar) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << "Opacity field index " << var_index << " is outside [0," << nVar - 1
        << "]." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (!std::isfinite(density) || !std::isfinite(temperature) ||
      density <= 0.0 || temperature <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << "Density and temperature must be finite and positive. Received rho=" << density
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
  if (!std::isfinite(log_temperature) || !std::isfinite(log_x2)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << "Opacity coordinates must be finite after conversion. Received rho="
        << density << ", T=" << temperature << ", resulting log10(T)="
        << log_temperature << ", x2=" << log_x2 << "." << std::endl;
    ATHENA_ERROR(msg);
  }

  const bool outside = log_temperature < tempMin || log_temperature > tempMax ||
                       log_x2 < pressureMin || log_x2 > pressureMax;
  if (outside) {
    out_of_domain_count_.fetch_add(1, std::memory_order_relaxed);
    if (domain_policy == DomainPolicy::error) {
      std::stringstream msg;
      msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
          << OpacityFieldName(var_index) << " opacity lookup is outside the table domain: "
          << "log10(T)=" << log_temperature << " (valid [" << tempMin << ","
          << tempMax << "]), x2=" << log_x2 << " (valid [" << pressureMin << ","
          << pressureMax << "]). fld/opacity_table_domain_policy=error."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    if (domain_policy == DomainPolicy::clamp) {
      log_temperature = std::max(tempMin, std::min(tempMax, log_temperature));
      log_x2 = std::max(pressureMin, std::min(pressureMax, log_x2));
      clamped_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  Real result = interpolate(var_index, log_x2, log_temperature);
  if (!std::isfinite(result)) {
    nonfinite_result_count_.fetch_add(1, std::memory_order_relaxed);
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << OpacityFieldName(var_index) << " opacity interpolation returned a non-finite "
        << "stored value at x2=" << log_x2 << ", log10(T)=" << log_temperature
        << "." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (values_are_log10[var_index]) {
    result = std::pow(static_cast<Real>(10.0), result);
  }
  if (!std::isfinite(result)) {
    nonfinite_result_count_.fetch_add(1, std::memory_order_relaxed);
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << OpacityFieldName(var_index) << " opacity is non-finite after decoding at x2="
        << log_x2 << ", log10(T)=" << log_temperature << "." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (result < 0.0) {
    const std::uint64_t negative_count =
        negative_result_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << OpacityFieldName(var_index) << " opacity extrapolation returned a negative "
        << "value " << result << " at x2=" << log_x2 << ", log10(T)="
        << log_temperature << ". Negative results are not replaced by a floor. "
        << "Local negative-result count=" << negative_count << "." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (result == 0.0) {
    zero_result_count_.fetch_add(1, std::memory_order_relaxed);
    std::stringstream msg;
    msg << "### FATAL ERROR in UserOpacityTable::GetOpacityFromRhoT" << std::endl
        << OpacityFieldName(var_index) << " opacity is zero at x2=" << log_x2
        << ", log10(T)=" << log_temperature
        << ". Opacity must be strictly positive." << std::endl;
    ATHENA_ERROR(msg);
  }
  return result;
}

UserOpacityTable::Diagnostics UserOpacityTable::GetLocalDiagnostics() const {
  Diagnostics diagnostics;
  diagnostics.out_of_domain = out_of_domain_count_.load(std::memory_order_relaxed);
  diagnostics.clamped = clamped_count_.load(std::memory_order_relaxed);
  diagnostics.negative_results = negative_result_count_.load(std::memory_order_relaxed);
  diagnostics.zero_results = zero_result_count_.load(std::memory_order_relaxed);
  diagnostics.nonfinite_results = nonfinite_result_count_.load(std::memory_order_relaxed);
  return diagnostics;
}

void UserOpacityTable::ReportDiagnostics(std::ostream &stream) const {
  const Diagnostics local = GetLocalDiagnostics();
  unsigned long long counts[5] = {
      static_cast<unsigned long long>(local.out_of_domain),
      static_cast<unsigned long long>(local.clamped),
      static_cast<unsigned long long>(local.negative_results),
      static_cast<unsigned long long>(local.zero_results),
      static_cast<unsigned long long>(local.nonfinite_results)};
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, counts, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (Globals::my_rank == 0) {
    stream << "FLD_OPACITY_DIAGNOSTICS out_of_domain=" << counts[0]
           << " clamped=" << counts[1]
           << " negative_results=" << counts[2]
           << " zero_results=" << counts[3]
           << " nonfinite_results=" << counts[4] << std::endl;
  }
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
