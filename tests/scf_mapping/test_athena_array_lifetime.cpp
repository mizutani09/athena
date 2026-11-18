#include <cassert>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../src/athena_arrays.hpp"

namespace {
using Array = AthenaArray<double>;

template <typename Constructor>
void CheckEmptyConstruction(Constructor construct) {
  typename std::aligned_storage<sizeof(Array), alignof(Array)>::type storage;
  std::memset(&storage, 0xa5, sizeof(storage));
  Array* result = construct(&storage);
  assert(result->IsEmpty());
  assert(result->data() == nullptr);
  assert(result->GetSize() == 0);
  result->~Array();
}

struct Holder {
  Array populated;
  Array empty;
};
}  // namespace

int main() {
  Array copy_source;
  CheckEmptyConstruction([&copy_source](void* storage) {
    return new (storage) Array(copy_source);
  });

  Array move_source;
  CheckEmptyConstruction([&move_source](void* storage) {
    return new (storage) Array(std::move(move_source));
  });

  // Exercise the MGOctet failure mode: vector growth relocates objects that
  // contain both allocated and empty AthenaArray members.
  std::vector<Holder> holders;
  holders.resize(1);
  holders.front().populated.NewAthenaArray(8);
  for (int size = 2; size <= 1024; size *= 2) holders.resize(size);
  for (Holder& holder : holders) {
    assert(holder.empty.IsEmpty());
    assert(holder.empty.data() == nullptr);
  }
  return 0;
}
