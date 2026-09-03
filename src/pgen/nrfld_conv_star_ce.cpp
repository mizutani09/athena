//========================================================================================
//! \file nrfld_conv_star_ce.cpp
//! \brief NR-FLD RSG plus a softened companion on a prescribed CE orbit.
//========================================================================================

// Keep the relaxed stellar model and radiation treatment identical to
// nrfld_conv_star, while selecting the CE-only companion source and diagnostics.
#define NRFLD_CONV_STAR_CE 1
#include "nrfld_conv_star.cpp"
