//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlld_fld.cpp
//! \brief HLLD with scalar FLD radiation pressure and radiation-energy flux splitting.

// The implementation is shared with hlld.cpp under NRMGFLD_ENABLED so that the
// well-tested MHD wave construction remains identical when radiation is disabled.
// Keep this wrapper as the configure-visible solver translation unit.
// Radiation coupling is restricted to active transverse control-volume faces.
// Non-finite diagnostics report both gas/MHD and reconstructed radiation states.
#include "hlld.cpp"  // NOLINT(build/include)
