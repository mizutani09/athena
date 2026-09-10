//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hdf5_reader.cpp
//! \brief Implements HDF5 reader functions

// C headers

// C++ headers
#include <iostream>   // cout
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../athena.hpp"         // Real
#include "../athena_arrays.hpp"  // AthenaArray
#include "../defs.hpp"           // SINGLE_PRECISION_ENABLED
#include "hdf5_reader.hpp"

// Only proceed if HDF5 enabled
#ifdef HDF5OUTPUT

// External library headers
#include <hdf5.h>  // H5[F|P|S|T]_*, H5[D|F|P|S]*(), hid_t
#ifdef MPI_PARALLEL
#include <mpi.h>  // MPI_COMM_WORLD, MPI_INFO_NULL
#endif

// Determine floating-point precision (in memory, not file)
#if SINGLE_PRECISION_ENABLED
#define H5T_REAL H5T_NATIVE_FLOAT
#else
#define H5T_REAL H5T_NATIVE_DOUBLE
#endif

namespace {
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

void HDF5ReadError(const char *filename, const char *dataset_name,
                   const std::string &operation) {
  std::stringstream msg;
  msg << "### FATAL ERROR in HDF5ReadRealArray" << std::endl
      << operation << " for HDF5 dataset '" << dataset_name << "' in file '"
      << filename << "'." << std::endl;
  ATHENA_ERROR(msg);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void HDF5ReadArray(const char *filename, const char *dataset_name, int rank_file,
//!     const int *start_file, const int *count_file, int rank_mem, const int *start_mem,
//!     const int *count_mem, AthenaArray<Real> &array, bool collective=false,
//!     bool noop=false)
//! \brief Read a single dataset from an HDF5 file into a pre-allocated array.

void HDF5ReadRealArray(const char *filename, const char *dataset_name, int rank_file,
                       const int *start_file, const int *count_file, int rank_mem,
                       const int *start_mem, const int *count_mem,
                       AthenaArray<Real> &array,
                       bool collective, bool noop) {
  // Check that user is not trying to exceed limits of HDF5 array or AthenaArray
  // dimensionality
  if (rank_file > MAX_RANK_FILE) {
    std::stringstream msg;
    msg << "### FATAL ERROR\nAttempting to read HDF5 array of ndim= " << rank_file
        << "\nExceeding MAX_RANK_FILE=" << MAX_RANK_FILE << std::endl;
    ATHENA_ERROR(msg);
  }
  if (rank_mem > MAX_RANK_MEM) {
    std::stringstream msg;
    msg << "### FATAL ERROR\nAttempting to read HDF5 array of ndim= " << rank_mem
        << "\nExceeding MAX_RANK_MEM=" << MAX_RANK_MEM << std::endl;
    ATHENA_ERROR(msg);
  }

  // Cast selection arrays to appropriate types
  hsize_t start_file_hid[MAX_RANK_FILE];
  hsize_t count_file_hid[MAX_RANK_FILE];
  for (int n = 0; n < rank_file; ++n) {
    start_file_hid[n] = start_file[n];
    count_file_hid[n] = count_file[n];
  }
  hsize_t start_mem_hid[MAX_RANK_MEM];
  hsize_t count_mem_hid[MAX_RANK_MEM];
  for (int n = 0; n < rank_mem; ++n) {
    start_mem_hid[n] = start_mem[n];
    count_mem_hid[n] = count_mem[n];
  }

  // Determine AthenaArray dimensions
  hsize_t dims_mem_base[5];
  dims_mem_base[0] = array.GetDim5();
  dims_mem_base[1] = array.GetDim4();
  dims_mem_base[2] = array.GetDim3();
  dims_mem_base[3] = array.GetDim2();
  dims_mem_base[4] = array.GetDim1();
  hsize_t *dims_mem = dims_mem_base + 5 - rank_mem;

  // Open data file
  HDF5Handle property_list_file(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
  if (!property_list_file.valid()) {
    HDF5ReadError(filename, dataset_name, "Could not create a file-access property list");
  }
#ifdef MPI_PARALLEL
  {
    if (collective) {
      if (H5Pset_fapl_mpio(property_list_file, MPI_COMM_WORLD, MPI_INFO_NULL) < 0) {
        HDF5ReadError(filename, dataset_name,
                      "Could not configure collective file access");
      }
    }
  }
#endif
  HDF5Handle file(H5Fopen(filename, H5F_ACC_RDONLY, property_list_file), H5Fclose);
  if (!file.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in HDF5ReadRealArray" << std::endl
        << "Could not open HDF5 file '" << filename << "' while reading dataset '"
        << dataset_name << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle property_list_transfer(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
  if (!property_list_transfer.valid()) {
    HDF5ReadError(filename, dataset_name,
                  "Could not create a dataset-transfer property list");
  }
#ifdef MPI_PARALLEL
  {
    if (collective) {
      if (H5Pset_dxpl_mpio(property_list_transfer, H5FD_MPIO_COLLECTIVE) < 0) {
        HDF5ReadError(filename, dataset_name,
                      "Could not configure collective dataset transfer");
      }
    }
  }
#endif

  // Read dataset into array
  HDF5Handle dataset(H5Dopen(file, dataset_name, H5P_DEFAULT), H5Dclose);
  if (!dataset.valid()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in HDF5ReadRealArray" << std::endl
        << "Could not open HDF5 dataset '" << dataset_name << "' in file '"
        << filename << "'." << std::endl;
    ATHENA_ERROR(msg);
  }
  HDF5Handle dataspace_file(H5Dget_space(dataset), H5Sclose);
  if (!dataspace_file.valid()) {
    HDF5ReadError(filename, dataset_name, "Could not get the file dataspace");
  }
  if (noop) {
    if (H5Sselect_none(dataspace_file) < 0) {
      HDF5ReadError(filename, dataset_name, "Could not select an empty file region");
    }
  } else if (H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start_file_hid, NULL,
                                 count_file_hid, NULL) < 0) {
    HDF5ReadError(filename, dataset_name, "Could not select the requested file region");
  }
  HDF5Handle dataspace_mem(H5Screate_simple(rank_mem, dims_mem, NULL), H5Sclose);
  if (!dataspace_mem.valid()) {
    HDF5ReadError(filename, dataset_name, "Could not create the memory dataspace");
  }
  if (noop) {
    if (H5Sselect_none(dataspace_mem) < 0) {
      HDF5ReadError(filename, dataset_name, "Could not select an empty memory region");
    }
  } else if (H5Sselect_hyperslab(dataspace_mem, H5S_SELECT_SET, start_mem_hid, NULL,
                                 count_mem_hid, NULL) < 0) {
    HDF5ReadError(filename, dataset_name, "Could not select the requested memory region");
  }
  if (H5Dread(dataset, H5T_REAL, dataspace_mem, dataspace_file, property_list_transfer,
              array.data()) < 0) {
    HDF5ReadError(filename, dataset_name, "Could not read data");
  }
}


//----------------------------------------------------------------------------------------
//! \fn void HDF5TableLoader(const char *filename, InterpTable2D* ptable, const int nvar,
//!                    const char **var_names, char *x2lim_name, char *x1lim_name) {
//! \brief Reads datasets from an HDF5 file into a InterpTable2D.

void HDF5TableLoader(const char *filename, InterpTable2D* ptable, const int nvar,
                     const char **var_names, const char *x2lim_name,
                     const char *x1lim_name) {
  if (nvar < 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
        << "At least one data field is required for HDF5 file '" << filename << "'."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  hsize_t dims[2];
  int tmp[2];
  int count_file[2];
  {
    HDF5Handle property_list_file(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    if (!property_list_file.valid()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
          << "Could not create a file-access property list for HDF5 file '"
          << filename << "'." << std::endl;
      ATHENA_ERROR(msg);
    }
    HDF5Handle file(H5Fopen(filename, H5F_ACC_RDONLY, property_list_file), H5Fclose);
    if (!file.valid()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
          << "Could not open HDF5 file '" << filename << "'." << std::endl;
      ATHENA_ERROR(msg);
    }
    for (int i = 0; i < nvar; ++i) {
      HDF5Handle dataset(H5Dopen(file, var_names[i], H5P_DEFAULT), H5Dclose);
      if (!dataset.valid()) {
        std::stringstream msg;
        msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
            << "Could not open HDF5 dataset '" << var_names[i] << "' in file '"
            << filename << "'." << std::endl;
        ATHENA_ERROR(msg);
      }
      HDF5Handle dspace(H5Dget_space(dataset), H5Sclose);
      if (!dspace.valid()) {
        std::stringstream msg;
        msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
            << "Could not get the dataspace for HDF5 dataset '" << var_names[i]
            << "' in file '" << filename << "'." << std::endl;
        ATHENA_ERROR(msg);
      }
      int ndims = H5Sget_simple_extent_ndims(dspace);
      if (ndims < 0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
            << "Could not read the rank of HDF5 dataset '" << var_names[i]
            << "' in file '" << filename << "'." << std::endl;
        ATHENA_ERROR(msg);
      }
      if (ndims != 2) {
        std::stringstream msg;
        msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
            << "Rank of data field '" << var_names[i] << "' in file '" << filename
            << "' must be 2. Rank is " << ndims << "." << std::endl;
        ATHENA_ERROR(msg);
      }
      if (H5Sget_simple_extent_dims(dspace, dims, NULL) < 0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
            << "Could not read the shape of HDF5 dataset '" << var_names[i]
            << "' in file '" << filename << "'." << std::endl;
        ATHENA_ERROR(msg);
      }
      tmp[0] = static_cast<int>(dims[0]);
      tmp[1] = static_cast<int>(dims[1]);
      if (i == 0) {
        count_file[0] = tmp[0];
        count_file[1] = tmp[1];
      } else if (count_file[0]!=tmp[0] || count_file[1]!=tmp[1]) {
        std::stringstream msg;
        msg << "### FATAL ERROR in HDF5TableLoader" << std::endl
            << "Inconsistent data field shape in file '" << filename << "'."
            << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }
  ptable->SetSize(nvar, count_file[0], count_file[1]);
  int start_file[2];
  start_file[0] = 0;
  start_file[1] = 0;
  int start_mem[3];
  start_mem[1] = 0;
  start_mem[2] = 0;
  int count_mem[3];
  count_mem[0] = 1;
  count_mem[1] = count_file[0];
  count_mem[2] = count_file[1];
  for (int i = 0; i < nvar; ++i) {
    start_mem[0] = i;
    HDF5ReadRealArray(filename, var_names[i], 2, start_file, count_file,
                      3, start_mem, count_mem, ptable->data);
  }
  if (x2lim_name) {
    AthenaArray<Real> lim;
    lim.NewAthenaArray(2);
    int zero[] = {0};
    int two[] = {2};
    HDF5ReadRealArray(filename, x2lim_name, 1, zero, two, 1, zero, two, lim);
    ptable->SetX2lim(lim(0), lim(1));
  }
  if (x1lim_name) {
    AthenaArray<Real> lim;
    lim.NewAthenaArray(2);
    int zero[] = {0};
    int two[] = {2};
    HDF5ReadRealArray(filename, x1lim_name, 1, zero, two, 1, zero, two, lim);
    ptable->SetX1lim(lim(0), lim(1));
  }
  return;
}
#endif  // HDF5OUTPUT
