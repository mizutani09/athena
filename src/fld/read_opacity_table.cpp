// //========================================================================================
// // Athena++ astrophysical MHD code
// // Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// // Licensed under the 3-clause BSD License, see LICENSE file for details
// //======================================================================================
// //! \file read_opacity_table.cpp
// //  \brief Implements class UserOpacityTable for an User-defined lookup table
// //======================================================================================

// // C headers

// // C++ headers
// #include <cmath>   // sqrt()
// #include <fstream>
// #include <iostream> // ifstream
// #include <sstream>
// #include <stdexcept> // std::invalid_argument
// #include <string>

// // Athena++ headers
// #include "../athena.hpp"
// #include "../athena_arrays.hpp"
// #include "../coordinates/coordinates.hpp"
// #include "../field/field.hpp"
// #include "../inputs/ascii_table_reader.hpp"
// #include "../inputs/hdf5_reader.hpp"
// #include "../parameter_input.hpp"
// #include "../utils/interp_table.hpp"
// #include "rad_fld.hpp"

// #ifdef HDF5OUTPUT
// #include <hdf5.h>
// #endif

// // Order of datafields for HDF5 opacity tables
// // These variable names should match the dataset names in your HDF5 opacity file
// // Order must match enum OpacityIndex {SIGMA_P=0, SIGMA_R=1} in rad_fld.hpp
// const char *opacity_var_names[] = {"planck_mean_opacity", "rosseland_mean_opacity"};

// //----------------------------------------------------------------------------------------
// //! \fn void ReadAsciiOpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin)
// //  \brief Read data from ascii opacity table and initialize interpolated table.
// void ReadAsciiOpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin) {
//   AthenaArray<Real> *ptables = nullptr;
//   if (puser_table->use_tables) ptables = &puser_table->OpacityTables;

//   // If use_tables then OpacityTables.NewAthenaArray is called in ASCIITableLoader
//   ASCIITableLoader(fn.c_str(), *puser_table, ptables);
//   puser_table->GetSize(puser_table->nVar, puser_table->nPressure, puser_table->nTemp);
//   puser_table->GetX2lim(puser_table->pressureMin, puser_table->pressureMax);
//   puser_table->GetX1lim(puser_table->tempMin, puser_table->tempMax);

//   if (!puser_table->use_tables) {
//     puser_table->OpacityTables.NewAthenaArray(puser_table->nVar);
//     for (int i=0; i<puser_table->nVar; ++i) puser_table->OpacityTables(i) = 1.0;
//   }
// }

// //----------------------------------------------------------------------------------------
// //! \fn void ReadHDF5OpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin)
// //  \brief Read data from HDF5 opacity table and initialize interpolated table.
// void ReadHDF5OpacityTable(std::string fn, UserOpacityTable *puser_table, ParameterInput *pin) {
//   #ifdef HDF5OUTPUT
//   // Get number of variables to read from the table
//   int nvar = RadFLD::NOPACITY;

//   // Get variable names - these should match the dataset names in your HDF5 file
//   const char **var_names = opacity_var_names;

//   // Read 2D grid format: separate 1D coordinate arrays and 2D opacity grids
//   AthenaArray<Real> temp_array, pressure_array;

//   // Read temperature coordinate array to get its size
//   int temp_size = 0;
//   {
//     hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
//     hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
//     hid_t dataset = H5Dopen(file, "log_temperature", H5P_DEFAULT);
//     hid_t dspace = H5Dget_space(dataset);
//     hsize_t dims[1];
//     H5Sget_simple_extent_dims(dspace, dims, NULL);
//     temp_size = static_cast<int>(dims[0]);
//     H5Sclose(dspace);
//     H5Dclose(dataset);
//     H5Fclose(file);
//     H5Pclose(property_list_file);
//   }

//   // Read pressure coordinate array to get its size
//   int pressure_size = 0;
//   {
//     hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
//     hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
//     hid_t dataset = H5Dopen(file, "log_pressure", H5P_DEFAULT);
//     hid_t dspace = H5Dget_space(dataset);
//     hsize_t dims[1];
//     H5Sget_simple_extent_dims(dspace, dims, NULL);
//     pressure_size = static_cast<int>(dims[0]);
//     H5Sclose(dspace);
//     H5Dclose(dataset);
//     H5Fclose(file);
//     H5Pclose(property_list_file);
//   }

//   // Set up proper 2D table structure
//   puser_table->SetSize(nvar, pressure_size, temp_size);  // nvar, nx2=pressure_size, nx1=temp_size
//   puser_table->nVar = nvar;
//   puser_table->nTemp = temp_size;
//   puser_table->nPressure = pressure_size;

//   // Read coordinate arrays
//   temp_array.NewAthenaArray(temp_size);
//   pressure_array.NewAthenaArray(pressure_size);

//   int start_file[1] = {0};
//   int start_mem[1] = {0};
//   int count_temp[1] = {temp_size};
//   int count_pressure[1] = {pressure_size};

//   HDF5ReadRealArray(fn.c_str(), "log_temperature", 1, start_file, count_temp,
//                     1, start_mem, count_temp, temp_array);
//   HDF5ReadRealArray(fn.c_str(), "log_pressure", 1, start_file, count_pressure,
//                     1, start_mem, count_pressure, pressure_array);

//   // Set coordinate limits in physical units
//   puser_table->tempMin = temp_array(0);
//   puser_table->tempMax = temp_array(temp_size - 1);
//   puser_table->pressureMin = pressure_array(0);
//   puser_table->pressureMax = pressure_array(pressure_size - 1);

//   // Set grid bounds in physical units for InterpTable2D
//   puser_table->SetX1lim(puser_table->tempMin, puser_table->tempMax);
//   puser_table->SetX2lim(puser_table->pressureMin, puser_table->pressureMax);

//   // Read 2D opacity data arrays
//   for (int ivar = 0; ivar < nvar; ++ivar) {
//     // Get opacity array dimensions to verify it's 2D
//     hsize_t opacity_dims[2];
//     {
//       hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
//       hid_t file = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, property_list_file);
//       hid_t dataset = H5Dopen(file, var_names[ivar], H5P_DEFAULT);
//       hid_t dspace = H5Dget_space(dataset);
//       int ndims = H5Sget_simple_extent_ndims(dspace);
//       if (ndims != 2) {
//         std::stringstream msg;
//         msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
//             << "Opacity dataset '" << var_names[ivar] << "' must be 2D, found " << ndims << "D" << std::endl;
//         ATHENA_ERROR(msg);
//       }
//       H5Sget_simple_extent_dims(dspace, opacity_dims, NULL);
//       H5Sclose(dspace);
//       H5Dclose(dataset);
//       H5Fclose(file);
//       H5Pclose(property_list_file);
//     }

//     // Verify dimensions match coordinate arrays
//     if (static_cast<int>(opacity_dims[0]) != pressure_size ||
//         static_cast<int>(opacity_dims[1]) != temp_size) {
//       std::stringstream msg;
//       msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
//           << "Opacity array dimensions [" << opacity_dims[0] << "," << opacity_dims[1]
//           << "] must match coordinate arrays [" << pressure_size << "," << temp_size << "]" << std::endl;
//       ATHENA_ERROR(msg);
//     }

//     // Read 2D opacity data
//     AthenaArray<Real> opacity_2d;
//     opacity_2d.NewAthenaArray(pressure_size, temp_size);

//     int start_file_2d[2] = {0, 0};
//     int start_mem_2d[2] = {0, 0};
//     int count_2d[2] = {pressure_size, temp_size};

//     HDF5ReadRealArray(fn.c_str(), var_names[ivar], 2, start_file_2d, count_2d,
//                       2, start_mem_2d, count_2d, opacity_2d);

//     // Store 2D opacity data with unit conversion from physical units (cm^2/g) to code units
//     // Note: HDF5 data is opacity[pressure_idx, temp_idx], InterpTable2D expects data(ivar, j, i)
//     for (int j = 0; j < pressure_size; ++j) {
//       for (int i = 0; i < temp_size; ++i) {
//         puser_table->data(ivar, j, i) = opacity_2d(j, i);
//       }
//     }

//     opacity_2d.DeleteAthenaArray();
//   }

//   // Clean up coordinate arrays
//   temp_array.DeleteAthenaArray();
//   pressure_array.DeleteAthenaArray();

//   if (!puser_table->use_tables) {
//     puser_table->OpacityTables.NewAthenaArray(puser_table->nVar);
//     for (int i=0; i<puser_table->nVar; ++i) puser_table->OpacityTables(i) = 1.0;
//   }
//   #else
//   std::stringstream msg;
//   msg << "### FATAL ERROR in ReadHDF5OpacityTable" << std::endl
//       << "HDF5 support not enabled. Reconfigure with -hdf5." << std::endl;
//   ATHENA_ERROR(msg);
//   #endif
// }

// //----------------------------------------------------------------------------------------
// //! \fn UserOpacityTable::UserOpacityTable(ParameterInput *pin)
// //  \brief Constructor for UserOpacityTable class, reads the opacity table from file.
// //   \param pin Pointer to ParameterInput object containing user-defined parameters.
// //   \note The table is read from a file specified in the parameter input, and the size of
// //         the table is set based on the data read.
// UserOpacityTable::UserOpacityTable(ParameterInput *pin) : InterpTable2D() {
//   use_tables = pin->GetOrAddBoolean("mgfld", "use_opacity_table", false);
//   if (!use_tables) return;
//   std::string opacity_fn, opacity_file_type;

//   // Get file name and type from parameters
//   opacity_fn = pin->GetString("mgfld", "opacity_table_file");
//   opacity_file_type = pin->GetOrAddString("mgfld", "opacity_table_file_type", "auto");

//   // Auto-detect file type based on extension if not specified
//   if (opacity_file_type.compare("auto") == 0) {
//     std::string ext = opacity_fn.substr(opacity_fn.find_last_of(".") + 1);
//     if (ext.compare("hdf5")*ext.compare("h5") == 0) {
//       opacity_file_type.assign("hdf5");
//     } else if (ext.compare("tab")*ext.compare("txt")*ext.compare("ascii") == 0) {
//       opacity_file_type.assign("ascii");
//     } else {
//       std::stringstream msg;
//       msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
//           << "Cannot auto-detect file type for '" << opacity_fn << "'." << std::endl
//           << "Please specify opacity_table_file_type explicitly." << std::endl;
//       ATHENA_ERROR(msg);
//     }
//   }

//   // Read the table based on file type
//   if (opacity_file_type.compare("hdf5") == 0) { // HDF5 table
//     #ifdef HDF5OUTPUT
//     ReadHDF5OpacityTable(opacity_fn, this, pin);
//     #else
//     std::stringstream msg;
//     msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
//         << "HDF5 support not enabled. Reconfigure with -hdf5." << std::endl;
//     ATHENA_ERROR(msg);
//     #endif
//   } else if (opacity_file_type.compare("ascii") == 0) { // ASCII/text table
//     ReadAsciiOpacityTable(opacity_fn, this, pin);
//   } else {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in UserOpacityTable::UserOpacityTable" << std::endl
//         << "Opacity table of type '" << opacity_file_type << "' not recognized."  << std::endl
//         << "Options are 'ascii' and 'hdf5'." << std::endl;
//     ATHENA_ERROR(msg);
//   }
// }

// //----------------------------------------------------------------------------------------
// //! \fn Real UserOpacityTable::GetOpacity(int var_index, Real x2, Real x1)
// //  \brief Gets interpolated opacity data from the 2D table using bilinear interpolation.
// //   \param var_index Index of the opacity variable
// //   \param x2 Second coordinate (pressure in physical units)
// //   \param x1 First coordinate (temperature in physical units)
// //   \note x2 and x1 are expected to be in physical units (erg/cm^3 for pressure, K for temperature)
// //   \note The interpolation is done on a logarithmic scale for both pressure and temperature
// Real UserOpacityTable::GetOpacity(int var_index, Real x2, Real x1) {
//   // For log-scale tables: x2 and x1 are physical pressure (erg/cm^3) and temperature (K)
//   // Take logarithms and interpolate on the log grid

//   if (!use_tables) {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in UserOpacityTable::GetOpacity" << std::endl
//         << "Opacity tables are not enabled. Set 'mgfld/use_opacity_table' to true." << std::endl;
//     ATHENA_ERROR(msg);
//   }

//   Real log_pressure = std::log10(x2);
//   Real log_temperature = std::log10(x1);

//   // Interpolate and return the result (already in code units)
//   Real result = interpolate(var_index, log_pressure, log_temperature);
//   if (result < 0.0) {
//     result = TINY_NUMBER; // Ensure non-negative opacity
//     std::cerr << "Warning: Negative opacity value encountered. Returning TINY_NUMBER instead." << std::endl;
//     std::cerr << "log_pressure: " << log_pressure << ", log_temperature: " << log_temperature << std::endl;
//     std::cerr << "Interpolated result: " << result << std::endl;
//   }
//   return result;
// }

// //----------------------------------------------------------------------------------------
// // UserOpacityTable destructor
// UserOpacityTable::~UserOpacityTable() {
//   // AthenaArray destructor will handle cleanup automatically
//   if (OpacityTables.GetDim1() > 0) {
//     OpacityTables.DeleteAthenaArray();
//   }
//   // Note: temp_coords and pressure_coords are AthenaArray objects,
//   // their destructors will handle cleanup automatically
// }
