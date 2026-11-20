#!/usr/bin/env python3
"""One-cycle smoke gate for the collaborator collapse solver v2 candidate."""

import collapse_firstcore_v1_smoke as harness


# The release candidate retains the v1 physical input and tables.  Its source
# adds table-MHD energy closure and the opt-in whole-solve recovery.  Pin the
# source tree rather than a commit so documentation-only release commits do
# not invalidate the gate.
harness.SOURCE_TREE = "8c4a94cb1b53b0ce6a7b574371351007a98472cb"
harness.PRODUCTION_ENV = {
    **harness.PRODUCTION_ENV,
    "ATHENA_NRFLD_TANGENTIAL_FULL_SOLVE_RETRY": "1",
}


if __name__ == "__main__":
    harness.main()
