#include <cassert>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "athena_arrays.hpp"

int main() {
  using Array = AthenaArray<double>;
  Array empty;
  using Storage = typename std::aligned_storage<sizeof(Array), alignof(Array)>::type;

  Storage copy_storage;
  std::memset(&copy_storage, 0xa5, sizeof(copy_storage));
  Array *copy = new (&copy_storage) Array(empty);
  assert(copy->data() == nullptr && copy->IsEmpty());
  copy->~Array();

  Storage move_storage;
  std::memset(&move_storage, 0xa5, sizeof(move_storage));
  Array *moved = new (&move_storage) Array(std::move(empty));
  assert(moved->data() == nullptr && moved->IsEmpty());
  moved->~Array();

  struct OctetLike {
    Array used;
    Array optional_matrix;
  };
  std::vector<OctetLike> octets(1);
  octets[0].used.NewAthenaArray(8);
  for (int size = 2; size <= 128; size *= 2) {
    octets.resize(size);
    for (auto &octet : octets)
      assert(octet.optional_matrix.data() == nullptr
             && octet.optional_matrix.IsEmpty());
  }
}
