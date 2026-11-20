#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PROFILE [OUTPUT_DIR]" >&2
  exit 2
fi

profile=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
output_dir=${2:-/tmp/wor151_cpp_validation}
build_dir=/tmp/wor151_cpp_build
binary="$build_dir/analyze_rsg_profile"
cxx=${CXX:-g++}

mkdir -p -- "$build_dir" "$output_dir"
"$cxx" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
  "$repo_dir/tools/mesa/analyze_rsg_profile.cpp" -o "$binary"

"$binary" --self-test
"$binary" --reference-gate "$profile"
"$binary" "$profile" --output-dir "$output_dir"
"$binary" --sensitivity "$profile" "$output_dir/core_cut_sensitivity.csv"

echo "WOR-151 C++ validation passed"
echo "output_dir=$output_dir"
