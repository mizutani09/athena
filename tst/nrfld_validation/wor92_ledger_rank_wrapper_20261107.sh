#!/usr/bin/env bash
set -euo pipefail
rank_id=${OMPI_COMM_WORLD_RANK:?OMPI rank is required}
: "${WOR92_LEDGER_RANK_DIR:?rank output directory is required}"
: "${WOR92_ATHENA_BIN:?Athena binary is required}"
exec "$WOR92_ATHENA_BIN" "$@" > "$WOR92_LEDGER_RANK_DIR/rank.${rank_id}.log" 2>&1
