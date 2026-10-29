// Production entry point for the Linear WOR-13 st35gm03n07 specification.
// The implementation is shared with the separately retained ideal-gas
// commissioning problem, but this entry point enforces table EOS, an imported
// HSE model, physical opacity tables, and no silent EOS clamping.
#define RSG_PRODUCTION 1
#include "nrfld_star_in_box.cpp"
