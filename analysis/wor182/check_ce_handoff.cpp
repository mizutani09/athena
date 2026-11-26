// Validate the structural and precision contract of a WOR-182 CE handoff file.
#include <hdf5.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<hsize_t> DatasetShape(hid_t file, const char* name,
                                  H5T_class_t* type_class = nullptr,
                                  std::size_t* type_size = nullptr) {
  hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
  Require(dataset >= 0, std::string("missing dataset ") + name);
  hid_t space = H5Dget_space(dataset);
  Require(space >= 0, std::string("cannot inspect dataspace ") + name);
  const int rank = H5Sget_simple_extent_ndims(space);
  Require(rank >= 0, std::string("cannot inspect rank ") + name);
  std::vector<hsize_t> shape(static_cast<std::size_t>(rank));
  Require(H5Sget_simple_extent_dims(space, shape.data(), nullptr) >= 0,
          std::string("cannot inspect shape ") + name);
  hid_t type = H5Dget_type(dataset);
  Require(type >= 0, std::string("cannot inspect type ") + name);
  if (type_class != nullptr) *type_class = H5Tget_class(type);
  if (type_size != nullptr) *type_size = H5Tget_size(type);
  H5Tclose(type);
  H5Sclose(space);
  H5Dclose(dataset);
  return shape;
}

int ReadScalarIntAttribute(hid_t file, const char* name) {
  hid_t attribute = H5Aopen(file, name, H5P_DEFAULT);
  Require(attribute >= 0, std::string("missing root attribute ") + name);
  int value = 0;
  Require(H5Aread(attribute, H5T_NATIVE_INT, &value) >= 0,
          std::string("cannot read root attribute ") + name);
  H5Aclose(attribute);
  return value;
}

std::string ShapeString(const std::vector<hsize_t>& shape) {
  std::string result = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) result += ",";
    result += std::to_string(shape[i]);
  }
  return result + "]";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: check_ce_handoff FILE [--require-float64]\n";
    return 2;
  }
  const bool require_float64 = argc == 3
      && std::string(argv[2]) == "--require-float64";
  if (argc == 3 && !require_float64) {
    std::cerr << "unknown option: " << argv[2] << "\n";
    return 2;
  }

  try {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t file = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    Require(file >= 0, std::string("cannot open ") + argv[1]);

    H5T_class_t cons_class = H5T_NO_CLASS;
    std::size_t cons_size = 0;
    const auto cons = DatasetShape(file, "/cons", &cons_class, &cons_size);
    const auto levels = DatasetShape(file, "/Levels");
    const auto locations = DatasetShape(file, "/LogicalLocations");
    const auto x1f = DatasetShape(file, "/x1f");
    const auto x2f = DatasetShape(file, "/x2f");
    const auto x3f = DatasetShape(file, "/x3f");

    Require(cons.size() == 5, "/cons must have rank 5");
    Require(cons_class == H5T_FLOAT, "/cons must use a floating-point type");
    Require(cons_size == 4 || cons_size == 8,
            "/cons must use float32 or float64 storage");
    Require(!require_float64 || cons_size == 8,
            "final CE handoff requires float64 /cons storage");
    Require(levels.size() == 1 && levels[0] == cons[1],
            "/Levels must contain one entry per MeshBlock");
    Require(locations.size() == 2 && locations[0] == cons[1]
                && locations[1] == 3,
            "/LogicalLocations must have shape [NumMeshBlocks,3]");
    Require(x1f.size() == 2 && x1f[0] == cons[1]
                && x1f[1] == cons[4] + 1,
            "/x1f shape is inconsistent with /cons");
    Require(x2f.size() == 2 && x2f[0] == cons[1]
                && x2f[1] == cons[3] + 1,
            "/x2f shape is inconsistent with /cons");
    Require(x3f.size() == 2 && x3f[0] == cons[1]
                && x3f[1] == cons[2] + 1,
            "/x3f shape is inconsistent with /cons");
    Require(ReadScalarIntAttribute(file, "NumMeshBlocks")
                == static_cast<int>(cons[1]),
            "NumMeshBlocks attribute is inconsistent with /cons");
    H5Fclose(file);

    std::cout << "cons_shape=" << ShapeString(cons)
              << " cons_precision=float" << 8*cons_size
              << " levels_shape=" << ShapeString(levels)
              << " logical_locations_shape=" << ShapeString(locations)
              << " face_shapes=" << ShapeString(x1f) << ","
              << ShapeString(x2f) << "," << ShapeString(x3f) << "\n";
  } catch (const std::exception& error) {
    std::cerr << "CE handoff validation failed: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
