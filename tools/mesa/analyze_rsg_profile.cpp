#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kG = 6.67430e-8;
constexpr double kC = 2.99792458e10;
constexpr double kMsun = 1.9884098706980504e33;
constexpr double kRsun = 6.957e10;
constexpr double kLsun = 3.828e33;

std::vector<std::string> Split(const std::string &line) {
  std::istringstream in(line);
  std::vector<std::string> words;
  for (std::string word; in >> word;) words.push_back(word);
  return words;
}

std::string TrimQuotes(std::string value) {
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::string JsonEscape(const std::string &value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

bool ParseDouble(const std::string &text, double *value) {
  try {
    std::size_t used = 0;
    *value = std::stod(text, &used);
    return used == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

// Small self-contained SHA-256 implementation. Keeping the digest in this
// executable makes provenance independent of shell utilities and path quoting.
class Sha256 {
 public:
  Sha256() { Reset(); }

  void Update(const unsigned char *data, std::size_t length) {
    for (std::size_t i = 0; i < length; ++i) {
      buffer_[buffer_size_++] = data[i];
      if (buffer_size_ == 64) {
        Transform(buffer_.data());
        bit_length_ += 512;
        buffer_size_ = 0;
      }
    }
  }

  std::string Final() {
    std::size_t i = buffer_size_;
    buffer_[i++] = 0x80;
    if (i > 56) {
      while (i < 64) buffer_[i++] = 0;
      Transform(buffer_.data());
      i = 0;
    }
    while (i < 56) buffer_[i++] = 0;
    bit_length_ += static_cast<std::uint64_t>(buffer_size_) * 8;
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer_[i++] = static_cast<unsigned char>((bit_length_ >> shift) & 0xff);
    }
    Transform(buffer_.data());
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t value : state_) out << std::setw(8) << value;
    return out.str();
  }

 private:
  static std::uint32_t RotateRight(std::uint32_t value, int amount) {
    return (value >> amount) | (value << (32 - amount));
  }

  void Reset() {
    state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    buffer_size_ = 0;
    bit_length_ = 0;
  }

  void Transform(const unsigned char *block) {
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
      const int j = i * 4;
      w[i] = (static_cast<std::uint32_t>(block[j]) << 24) |
             (static_cast<std::uint32_t>(block[j + 1]) << 16) |
             (static_cast<std::uint32_t>(block[j + 2]) << 8) |
             static_cast<std::uint32_t>(block[j + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
    std::uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t temp1 = h + s1 + choose + k[i] + w[i];
      const std::uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + majority;
      h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
    state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<unsigned char, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_length_ = 0;
};

std::string FileSha256(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open for SHA-256: " + path.string());
  Sha256 sha;
  std::array<char, 1 << 16> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    sha.Update(reinterpret_cast<const unsigned char *>(buffer.data()),
               static_cast<std::size_t>(input.gcount()));
  }
  return sha.Final();
}

struct HeaderValue {
  std::string raw;
  std::optional<double> number;
};

std::map<std::string, HeaderValue> ReadMesaHeader(const fs::path &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open MESA profile: " + path.string());
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line) && lines.size() < 3;) {
    if (line.find_first_not_of(" \t\r") != std::string::npos) lines.push_back(line);
  }
  if (lines.size() < 3) throw std::runtime_error("MESA profile header is too short");
  const auto names = Split(lines[1]);
  const auto values = Split(lines[2]);
  if (names.size() != values.size()) {
    throw std::runtime_error("MESA header name/value counts differ");
  }
  std::map<std::string, HeaderValue> header;
  for (std::size_t i = 0; i < names.size(); ++i) {
    HeaderValue value;
    value.raw = TrimQuotes(values[i]);
    double number = 0.0;
    if (ParseDouble(value.raw, &number)) value.number = number;
    header.emplace(names[i], value);
  }
  return header;
}

struct Profile {
  std::map<std::string, HeaderValue> header;
  std::map<std::string, std::vector<double>> columns;
  std::size_t zones = 0;
};

Profile ReadMesaProfile(const fs::path &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open MESA profile: " + path.string());
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    if (line.find_first_not_of(" \t\r") != std::string::npos) lines.push_back(line);
  }
  if (lines.size() < 6) throw std::runtime_error("MESA profile is too short");
  const auto header_names = Split(lines[1]);
  const auto header_values = Split(lines[2]);
  if (header_names.size() != header_values.size()) {
    throw std::runtime_error("MESA header name/value counts differ");
  }
  Profile profile;
  for (std::size_t i = 0; i < header_names.size(); ++i) {
    HeaderValue value;
    value.raw = TrimQuotes(header_values[i]);
    double number = 0.0;
    if (ParseDouble(value.raw, &number)) value.number = number;
    profile.header.emplace(header_names[i], value);
  }
  const auto names = Split(lines[4]);
  if (names.empty()) throw std::runtime_error("MESA profile has no column names");
  for (const auto &name : names) profile.columns[name] = {};
  for (std::size_t line_index = 5; line_index < lines.size(); ++line_index) {
    const auto words = Split(lines[line_index]);
    if (words.size() != names.size()) {
      throw std::runtime_error("MESA data column count differs at input line " +
                               std::to_string(line_index + 1));
    }
    for (std::size_t column = 0; column < names.size(); ++column) {
      double value = 0.0;
      if (!ParseDouble(words[column], &value)) {
        throw std::runtime_error("non-numeric MESA data value at input line " +
                                 std::to_string(line_index + 1));
      }
      profile.columns[names[column]].push_back(value);
    }
  }
  profile.zones = profile.columns.begin()->second.size();
  if (!profile.columns.count("mass")) throw std::runtime_error("MESA profile has no mass column");
  std::vector<std::size_t> order(profile.zones);
  std::iota(order.begin(), order.end(), 0);
  const auto &unsorted_mass = profile.columns.at("mass");
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return unsorted_mass[a] < unsorted_mass[b];
  });
  for (auto &[name, values] : profile.columns) {
    std::vector<double> sorted;
    sorted.reserve(values.size());
    for (std::size_t index : order) sorted.push_back(values[index]);
    values.swap(sorted);
  }
  const auto &mass = profile.columns.at("mass");
  for (std::size_t i = 1; i < mass.size(); ++i) {
    if (!(mass[i] > mass[i - 1])) throw std::runtime_error("mass coordinate is not strictly increasing");
  }
  const std::set<std::string> required = {
      "mass","luminosity","eps_nuc","gradr_sub_grada","gradr","grada",
      "pressure_scale_height","pgas_div_ptotal","csound","tau","brunt_N2",
      "entropy","h1","he4","mix_type"};
  for (const auto &name : required) {
    if (!profile.columns.count(name)) throw std::runtime_error("missing required MESA column: " + name);
  }
  for (const auto &[linear, logarithmic] : std::array<std::pair<std::string,std::string>,5>{
           {{"radius","logR"},{"pressure","logP"},{"temperature","logT"},
            {"opacity","log_opacity"},{"density","logRho"}}}) {
    if (!profile.columns.count(linear) && !profile.columns.count(logarithmic)) {
      throw std::runtime_error("missing MESA column: " + linear + " or " + logarithmic);
    }
  }
  return profile;
}

double HeaderNumber(const std::map<std::string, HeaderValue> &header,
                    const std::string &name) {
  const auto it = header.find(name);
  return it != header.end() && it->second.number
      ? *it->second.number : std::numeric_limits<double>::quiet_NaN();
}

const std::vector<double> &Column(const Profile &profile, std::string_view name) {
  auto it = profile.columns.find(std::string(name));
  if (it == profile.columns.end()) {
    throw std::runtime_error("missing column: " + std::string(name));
  }
  return it->second;
}

std::vector<double> LinearOrLog(const Profile &profile, const std::string &linear,
                                const std::string &logarithmic) {
  auto it = profile.columns.find(linear);
  if (it != profile.columns.end()) return it->second;
  std::vector<double> result;
  result.reserve(profile.zones);
  for (double value : Column(profile, logarithmic)) result.push_back(std::pow(10.0, value));
  return result;
}

double HeaderNumber(const Profile &profile, const std::string &name, double fallback) {
  auto it = profile.header.find(name);
  return it != profile.header.end() && it->second.number ? *it->second.number : fallback;
}

std::vector<std::pair<std::size_t,std::size_t>> TrueRuns(const std::vector<bool> &mask) {
  std::vector<std::pair<std::size_t,std::size_t>> result;
  for (std::size_t i = 0; i < mask.size();) {
    if (!mask[i]) { ++i; continue; }
    const std::size_t begin = i;
    while (i + 1 < mask.size() && mask[i + 1]) ++i;
    result.emplace_back(begin, i);
    ++i;
  }
  return result;
}

struct Boundary {
  double mass = 0.0;
  std::size_t region_begin = 0, region_end = 0;
  std::size_t bracket_begin = 0, bracket_end = 0;
  std::vector<std::pair<double,double>> all_regions;
};

Boundary FindConvectiveBoundary(const Profile &profile) {
  const auto &mass = Column(profile, "mass");
  const auto &mix = Column(profile, "mix_type");
  const auto &delta = Column(profile, "gradr_sub_grada");
  std::vector<bool> convective(profile.zones);
  for (std::size_t i = 0; i < profile.zones; ++i) convective[i] = std::llround(mix[i]) == 1;
  const auto runs = TrueRuns(convective);
  if (runs.empty()) throw std::runtime_error("no convective region found");
  const auto best = *std::max_element(runs.begin(), runs.end(), [&](auto a, auto b) {
    return mass[a.second] - mass[a.first] < mass[b.second] - mass[b.first];
  });
  Boundary result;
  result.region_begin = best.first;
  result.region_end = best.second;
  result.mass = mass[best.first];
  result.bracket_begin = best.first == 0 ? 0 : best.first - 1;
  result.bracket_end = best.first;
  const std::size_t begin = best.first > 2 ? best.first - 2 : 0;
  const std::size_t end = std::min(profile.zones - 1, best.first + 8);
  for (std::size_t i = begin; i < end; ++i) {
    if (delta[i] <= 0.0 && delta[i + 1] > 0.0) {
      const double fraction = -delta[i] / (delta[i + 1] - delta[i]);
      result.mass = mass[i] + fraction * (mass[i + 1] - mass[i]);
      result.bracket_begin = i;
      result.bracket_end = i + 1;
      break;
    }
  }
  for (auto run : runs) result.all_regions.emplace_back(mass[run.first], mass[run.second]);
  return result;
}

struct CoreCut {
  double mass = 0.0;
  std::size_t zone = 0;
  double formal_he_core_mass = std::numeric_limits<double>::quiet_NaN();
  double nuclear_peak_mass = 0.0;
  double nuclear_peak_eps = 0.0;
  bool fallback = false;
};

CoreCut FindCoreCut(const Profile &profile, double boundary_mass, double eps_fraction,
                    double luminosity_fraction, int stable_zones) {
  if (stable_zones <= 0) throw std::runtime_error("stable-zones must be positive");
  const auto &mass = Column(profile, "mass");
  const auto &eps_raw = Column(profile, "eps_nuc");
  const auto &luminosity = Column(profile, "luminosity");
  const auto &delta = Column(profile, "gradr_sub_grada");
  std::size_t peak_index = 0;
  double peak = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < profile.zones && mass[i] < boundary_mass; ++i) {
    const double value = std::max(eps_raw[i], 0.0);
    if (value > peak) { peak = value; peak_index = i; }
  }
  const double surface_luminosity = HeaderNumber(profile, "photosphere_L", luminosity.back());
  std::optional<std::size_t> selected;
  for (std::size_t i = peak_index + 1; i + static_cast<std::size_t>(stable_zones) <= profile.zones; ++i) {
    if (mass[i] >= boundary_mass) break;
    bool stable = true;
    for (int j = 0; j < stable_zones; ++j) stable = stable && delta[i + j] < 0.0;
    const bool quiet = std::max(eps_raw[i], 0.0) <= peak * eps_fraction;
    const bool settled = std::abs(luminosity[i] - surface_luminosity) /
                         std::max(std::abs(surface_luminosity), 1.0e-99) <= luminosity_fraction;
    if (stable && quiet && settled) { selected = i; break; }
  }
  CoreCut result;
  result.formal_he_core_mass = HeaderNumber(profile, "he_core_mass",
                                             std::numeric_limits<double>::quiet_NaN());
  result.nuclear_peak_mass = mass[peak_index];
  result.nuclear_peak_eps = peak;
  if (selected) {
    result.zone = *selected;
    result.mass = mass[*selected];
  } else {
    if (!std::isfinite(result.formal_he_core_mass)) {
      throw std::runtime_error("no operational core cut and no he_core_mass fallback");
    }
    result.zone = static_cast<std::size_t>(std::lower_bound(
        mass.begin(), mass.end(), result.formal_he_core_mass) - mass.begin());
    result.mass = result.formal_he_core_mass;
    result.fallback = true;
  }
  return result;
}

using Table = std::map<std::string, std::vector<double>>;

Table BuildDerived(const Profile &profile) {
  const auto &mass = Column(profile, "mass");
  const auto radius = LinearOrLog(profile, "radius", "logR");
  const auto rho = LinearOrLog(profile, "density", "logRho");
  const auto pressure = LinearOrLog(profile, "pressure", "logP");
  const auto temperature = LinearOrLog(profile, "temperature", "logT");
  const auto opacity = LinearOrLog(profile, "opacity", "log_opacity");
  const auto &hp_rsun = Column(profile, "pressure_scale_height");
  const auto &beta = Column(profile, "pgas_div_ptotal");
  const auto &csound = Column(profile, "csound");
  std::vector<double> hp(profile.zones), gamma1(profile.zones), prad(profile.zones);
  std::vector<double> t_dyn(profile.zones), t_diff(profile.zones), binding(profile.zones);
  std::vector<double> t_thermal(profile.zones), kappa_p(profile.zones,
                                std::numeric_limits<double>::quiet_NaN());
  std::vector<double> dm(profile.zones), shell_energy(profile.zones, 0.0);
  const auto total_energy_it = profile.columns.find("total_energy");
  for (std::size_t i = 0; i < profile.zones; ++i) {
    hp[i] = hp_rsun[i] * kRsun;
    gamma1[i] = csound[i] * csound[i] * rho[i] / pressure[i];
    prad[i] = 1.0 - beta[i];
    const double radius_cgs = radius[i] * kRsun;
    const double mass_cgs = mass[i] * kMsun;
    t_dyn[i] = std::sqrt(std::pow(radius_cgs, 3) / std::max(kG * mass_cgs, 1.0e-99));
    t_diff[i] = 3.0 * opacity[i] * rho[i] * hp[i] * hp[i] / kC;
    dm[i] = (i == 0 ? mass[i] : mass[i] - mass[i - 1]) * kMsun;
    if (total_energy_it != profile.columns.end()) shell_energy[i] = total_energy_it->second[i] * dm[i];
  }
  double exterior = 0.0;
  const double luminosity_ref = HeaderNumber(profile, "photosphere_L", Column(profile, "luminosity").back());
  for (std::size_t reverse = profile.zones; reverse-- > 0;) {
    exterior += shell_energy[reverse];
    binding[reverse] = std::max(-exterior, 0.0);
    t_thermal[reverse] = binding[reverse] / std::max(luminosity_ref * kLsun, 1.0e-99);
  }
  auto optional_column = [&](const std::string &name) {
    auto it = profile.columns.find(name);
    return it == profile.columns.end()
        ? std::vector<double>(profile.zones, std::numeric_limits<double>::quiet_NaN())
        : it->second;
  };
  return {
      {"mass_Msun",mass},{"radius_Rsun",radius},{"rho_g_cm3",rho},
      {"pressure_dyn_cm2",pressure},{"temperature_K",temperature},
      {"pressure_scale_height_cm",hp},{"gamma1",gamma1},{"prad_over_ptotal",prad},
      {"luminosity_Lsun",Column(profile,"luminosity")},
      {"eps_nuc_erg_g_s",Column(profile,"eps_nuc")},
      {"eps_nu_non_nuclear_erg_g_s",optional_column("non_nuc_neu")},
      {"kappa_R_cm2_g",opacity},{"kappa_P_cm2_g",kappa_p},
      {"optical_depth",Column(profile,"tau")},{"grad_rad",Column(profile,"gradr")},
      {"grad_ad",Column(profile,"grada")},{"grad_rad_minus_grad_ad",Column(profile,"gradr_sub_grada")},
      {"brunt_N2_s-2",Column(profile,"brunt_N2")},{"entropy_over_NA_kB",Column(profile,"entropy")},
      {"t_diff_local_s",t_diff},{"t_dyn_local_s",t_dyn},{"t_thermal_exterior_s",t_thermal},
      {"binding_energy_exterior_erg",binding},{"h1_mass_fraction",Column(profile,"h1")},
      {"he4_mass_fraction",Column(profile,"he4")},{"mix_type",Column(profile,"mix_type")}};
}

double Interpolate(const std::vector<double> &x, const std::vector<double> &y, double target) {
  if (target <= x.front()) return y.front();
  if (target >= x.back()) return y.back();
  auto upper = std::upper_bound(x.begin(), x.end(), target);
  const std::size_t hi = static_cast<std::size_t>(upper - x.begin());
  const std::size_t lo = hi - 1;
  const double fraction = (target - x[lo]) / (x[hi] - x[lo]);
  return y[lo] + fraction * (y[hi] - y[lo]);
}

void WriteCsv(const fs::path &path, const Table &table) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write CSV: " + path.string());
  bool first = true;
  for (const auto &[name, values] : table) { if (!first) out << ','; out << name; first = false; }
  out << '\n' << std::scientific << std::setprecision(16);
  const std::size_t rows = table.begin()->second.size();
  for (std::size_t row = 0; row < rows; ++row) {
    first = true;
    for (const auto &[name, values] : table) {
      if (!first) out << ',';
      out << values[row];
      first = false;
    }
    out << '\n';
  }
}

void JsonNumber(std::ostream &out, double value) {
  if (std::isfinite(value)) out << std::setprecision(17) << value;
  else out << "null";
}

void WriteState(std::ostream &out, const Table &table, double target, int indent) {
  const auto &mass = table.at("mass_Msun");
  bool first = true;
  out << "{\n";
  for (const auto &[name, values] : table) {
    if (!first) out << ",\n";
    out << std::string(indent + 2, ' ') << '"' << JsonEscape(name) << "\": ";
    JsonNumber(out, Interpolate(mass, values, target));
    first = false;
  }
  out << '\n' << std::string(indent, ' ') << '}';
}

void WriteMetadata(const fs::path &path, const fs::path &source, const std::string &sha,
                   const Profile &profile, const Boundary &boundary, const CoreCut &core,
                   const Table &table, double eps_fraction, double luminosity_fraction,
                   int stable_zones, double exterior_nuclear_luminosity) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write JSON: " + path.string());
  const auto &mass = table.at("mass_Msun");
  out << "{\n  \"schema_version\": 2,\n";
  out << "  \"implementation\": \"C++17\",\n";
  out << "  \"source\": {\n    \"path_as_run\": \"" << JsonEscape(source.string()) << "\",\n";
  out << "    \"sha256\": \"" << sha << "\",\n    \"zones\": " << profile.zones << ",\n";
  out << "    \"mesa_header\": {\n";
  bool first = true;
  for (const auto &[name, value] : profile.header) {
    if (!first) out << ",\n";
    out << "      \"" << JsonEscape(name) << "\": ";
    if (value.number) JsonNumber(out, *value.number);
    else out << '"' << JsonEscape(value.raw) << '"';
    first = false;
  }
  out << "\n    }\n  },\n  \"selection\": {\n";
  out << "    \"core_cut\": {\n      \"mass_Msun\": "; JsonNumber(out, core.mass);
  out << ",\n      \"zone_index\": " << core.zone << ",\n      \"formal_he_core_mass_Msun\": "; JsonNumber(out, core.formal_he_core_mass);
  out << ",\n      \"nuclear_peak_mass_Msun\": "; JsonNumber(out, core.nuclear_peak_mass);
  out << ",\n      \"nuclear_peak_eps_erg_g_s\": "; JsonNumber(out, core.nuclear_peak_eps);
  out << ",\n      \"fallback_to_formal_core\": " << (core.fallback ? "true" : "false") << ",\n";
  out << "      \"criterion\": {\n        \"outside_dominant_eps_nuc_peak\": true,\n";
  out << "        \"eps_nuc_over_peak_max\": "; JsonNumber(out, eps_fraction);
  out << ",\n        \"relative_L_minus_Lphot_max\": "; JsonNumber(out, luminosity_fraction);
  out << ",\n        \"successive_radiatively_stable_zones\": " << stable_zones << "\n      }\n    },\n";
  out << "    \"radiative_convective_boundary\": {\n      \"mass_Msun\": "; JsonNumber(out, boundary.mass);
  out << ",\n      \"region_index\": [" << boundary.region_begin << ", " << boundary.region_end << "],\n";
  out << "      \"bracket_index\": [" << boundary.bracket_begin << ", " << boundary.bracket_end << "],\n";
  out << "      \"criterion\": \"largest mix_type == 1 mass interval; zero refined with gradr-grada\",\n";
  out << "      \"all_convective_regions_Msun\": [";
  for (std::size_t i = 0; i < boundary.all_regions.size(); ++i) {
    if (i) out << ", ";
    out << '[';
    JsonNumber(out, boundary.all_regions[i].first);
    out << ", ";
    JsonNumber(out, boundary.all_regions[i].second); out << ']';
  }
  out << "]\n    },\n    \"radiative_layer_mass_Msun\": "; JsonNumber(out, boundary.mass - core.mass);
  out << ",\n    \"convective_layer_mass_Msun\": "; JsonNumber(out, mass.back() - boundary.mass);
  out << ",\n    \"nuclear_luminosity_exterior_to_cut_Lsun\": "; JsonNumber(out, exterior_nuclear_luminosity);
  out << ",\n    \"source_is_inside_radiative_layer\": " << (core.mass < boundary.mass ? "true" : "false") << ",\n";
  out << "    \"core_cut_state\": "; WriteState(out, table, core.mass, 4); out << ",\n";
  out << "    \"boundary_state\": "; WriteState(out, table, boundary.mass, 4); out << "\n  },\n";
  out << "  \"definitions\": {\n"
      << "    \"gamma1\": \"csound^2 * rho / Ptotal\",\n"
      << "    \"t_diff_local\": \"3 * kappa_R * rho * H_P^2 / c\",\n"
      << "    \"t_dyn_local\": \"sqrt(r^3 / (G m))\",\n"
      << "    \"binding_energy_exterior\": \"max(-integral_m^M total_energy dm, 0)\",\n"
      << "    \"t_thermal_exterior\": \"binding_energy_exterior / photosphere_L\",\n"
      << "    \"kappa_P\": \"not exported by this MESA profile; CSV values are NaN\"\n  },\n";
  out << "  \"units\": {\n"
      << "    \"mass\": \"Msun\", \"radius\": \"Rsun\", \"rho\": \"g cm^-3\",\n"
      << "    \"pressure\": \"dyn cm^-2\", \"temperature\": \"K\",\n"
      << "    \"pressure_scale_height\": \"cm\", \"luminosity\": \"Lsun\",\n"
      << "    \"eps_nuc_and_eps_nu\": \"erg g^-1 s^-1\", \"opacity\": \"cm^2 g^-1\",\n"
      << "    \"brunt_N2\": \"s^-2\", \"times\": \"s\", \"binding_energy\": \"erg\"\n  },\n";
  out << "  \"outputs\": {\"profile_csv\": \"mesa_rsg_layers.csv\", \"plot\": \"mesa_rsg_layers.svg\"}\n}\n";
}

struct Panel {
  double x=0, y=0, width=0, height=0;
};

std::pair<double,double> FiniteRange(const std::vector<double> &values, bool logarithmic,
                                     const std::vector<bool> *mask = nullptr) {
  double low = std::numeric_limits<double>::infinity();
  double high = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (mask && !(*mask)[i]) continue;
    double value = values[i];
    if (logarithmic) {
      if (!(value > 0.0)) continue;
      value = std::log10(value);
    }
    if (!std::isfinite(value)) continue;
    low = std::min(low, value); high = std::max(high, value);
  }
  if (!std::isfinite(low) || !(high > low)) return {0.0, 1.0};
  const double pad = 0.05 * (high - low);
  return {low - pad, high + pad};
}

void SvgPolyline(std::ostream &out, const Panel &panel, const std::vector<double> &x,
                 const std::vector<double> &y, double xmin, double xmax,
                 double ymin, double ymax, bool log_y, const std::string &color,
                 const std::vector<bool> *mask = nullptr) {
  out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"1.4\" points=\"";
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (mask && !(*mask)[i]) continue;
    double value = y[i];
    if (log_y) { if (!(value > 0.0)) continue; value = std::log10(value); }
    if (!std::isfinite(value) || x[i] < xmin || x[i] > xmax) continue;
    const double px = panel.x + (x[i] - xmin) / (xmax - xmin) * panel.width;
    const double py = panel.y + panel.height - (value - ymin) / (ymax - ymin) * panel.height;
    out << px << ',' << py << ' ';
  }
  out << "\"/>\n";
}

void WriteSvg(const fs::path &path, const Table &table, double core, double boundary) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write SVG: " + path.string());
  const auto &mass = table.at("mass_Msun");
  const std::array<Panel,4> panels = {{{70,55,470,300},{620,55,470,300},{70,430,470,300},{620,430,470,300}}};
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1160\" height=\"790\" viewBox=\"0 0 1160 790\">\n"
      << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n"
      << "<text x=\"580\" y=\"25\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"18\">WOR-151 MESA RSG layer selection (C++17)</text>\n";
  const double full_min = mass.front(), full_max = mass.back();
  const double margin = std::max(boundary - core, 0.1);
  const double local_min = std::max(full_min, core - 1.5 * margin);
  const double local_max = std::min(full_max, boundary + 2.0 * margin);
  std::vector<bool> local(mass.size());
  for (std::size_t i=0;i<mass.size();++i) local[i]=mass[i]>=local_min && mass[i]<=local_max;
  auto draw_frame = [&](const Panel &p, const std::string &title) {
    out << "<rect x=\""<<p.x<<"\" y=\""<<p.y<<"\" width=\""<<p.width<<"\" height=\""<<p.height<<"\" fill=\"none\" stroke=\"#333\"/>\n";
    out << "<text x=\""<<p.x+p.width/2<<"\" y=\""<<p.y-10<<"\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">"<<title<<"</text>\n";
    out << "<text x=\""<<p.x+p.width/2<<"\" y=\""<<p.y+p.height+28<<"\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">mass coordinate [Msun]</text>\n";
  };
  auto markers = [&](const Panel &p, double xmin, double xmax) {
    for (auto [value,color] : std::array<std::pair<double,std::string>,2>{{{core,"#d62728"},{boundary,"#2ca02c"}}}) {
      if (value < xmin || value > xmax) continue;
      const double px=p.x+(value-xmin)/(xmax-xmin)*p.width;
      out << "<line x1=\""<<px<<"\" y1=\""<<p.y<<"\" x2=\""<<px<<"\" y2=\""<<p.y+p.height<<"\" stroke=\""<<color<<"\" stroke-dasharray=\"5,4\"/>\n";
    }
  };
  draw_frame(panels[0],"entropy / (N_A k_B)");
  auto yr=FiniteRange(table.at("entropy_over_NA_kB"),false);
  SvgPolyline(out,panels[0],mass,table.at("entropy_over_NA_kB"),full_min,full_max,yr.first,yr.second,false,"#1f77b4"); markers(panels[0],full_min,full_max);
  draw_frame(panels[1],"temperature gradients near retained intershell");
  auto y1=FiniteRange(table.at("grad_rad"),false,&local); auto y2=FiniteRange(table.at("grad_ad"),false,&local);
  const double gymin=std::min(y1.first,y2.first), gymax=std::max(y1.second,y2.second);
  SvgPolyline(out,panels[1],mass,table.at("grad_rad"),local_min,local_max,gymin,gymax,false,"#1f77b4",&local);
  SvgPolyline(out,panels[1],mass,table.at("grad_ad"),local_min,local_max,gymin,gymax,false,"#ff7f0e",&local); markers(panels[1],local_min,local_max);
  draw_frame(panels[2],"nuclear rate (blue) and luminosity (orange), log scale");
  auto e=FiniteRange(table.at("eps_nuc_erg_g_s"),true); auto l=FiniteRange(table.at("luminosity_Lsun"),true);
  const double lymin=std::min(e.first,l.first), lymax=std::max(e.second,l.second);
  SvgPolyline(out,panels[2],mass,table.at("eps_nuc_erg_g_s"),full_min,full_max,lymin,lymax,true,"#1f77b4");
  SvgPolyline(out,panels[2],mass,table.at("luminosity_Lsun"),full_min,full_max,lymin,lymax,true,"#ff7f0e"); markers(panels[2],full_min,full_max);
  draw_frame(panels[3],"local timescales [s], log scale");
  auto td=FiniteRange(table.at("t_dyn_local_s"),true); auto tf=FiniteRange(table.at("t_diff_local_s"),true); auto tt=FiniteRange(table.at("t_thermal_exterior_s"),true);
  const double tymin=std::min({td.first,tf.first,tt.first}), tymax=std::max({td.second,tf.second,tt.second});
  SvgPolyline(out,panels[3],mass,table.at("t_dyn_local_s"),full_min,full_max,tymin,tymax,true,"#1f77b4");
  SvgPolyline(out,panels[3],mass,table.at("t_diff_local_s"),full_min,full_max,tymin,tymax,true,"#ff7f0e");
  SvgPolyline(out,panels[3],mass,table.at("t_thermal_exterior_s"),full_min,full_max,tymin,tymax,true,"#9467bd"); markers(panels[3],full_min,full_max);
  out << "<text x=\"580\" y=\"775\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\"><tspan fill=\"#d62728\">-- M_c</tspan><tspan dx=\"25\" fill=\"#2ca02c\">-- M_b</tspan></text>\n</svg>\n";
}

Profile SyntheticProfile() {
  Profile p;
  p.zones=10;
  p.header["photosphere_L"]={"100",100.0};
  p.header["he_core_mass"]={"3",3.0};
  p.columns["mass"]={1,2,3,4,5,6,7,8,9,10};
  p.columns["mix_type"]={1,1,0,0,0,0,1,1,1,0};
  p.columns["gradr_sub_grada"]={1,1,-1,-1,-1,-0.5,-0.1,0.3,1,-1};
  p.columns["eps_nuc"]={0,1,100,10,0.001,0.001,0,0,0,0};
  p.columns["luminosity"]={1,2,80,98,99.9,100,100,100,100,100};
  return p;
}

void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error("self-test failed: " + message);
}

void SelfTest() {
  Require(TrueRuns({false,true,true,false,true}) ==
          std::vector<std::pair<std::size_t,std::size_t>>{{1,2},{4,4}}, "TrueRuns");
  Profile p=SyntheticProfile();
  Boundary b=FindConvectiveBoundary(p);
  Require(std::abs(b.mass-7.25)<1e-12 && b.region_begin==6 && b.region_end==8,
          "outer convective envelope selection");
  p.columns["gradr_sub_grada"]={-1,-1,-1,-1,-1,-1,1,1,1,-1};
  CoreCut c=FindCoreCut(p,7.0,1e-4,2e-3,2);
  Require(std::abs(c.mass-5.0)<1e-12 && !c.fallback, "operational core cut");
  Sha256 sha; const std::string abc="abc";
  sha.Update(reinterpret_cast<const unsigned char*>(abc.data()),abc.size());
  Require(sha.Final()=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256");
  std::cout << "all C++ self-tests passed\n";
}

struct Options {
  fs::path profile;
  fs::path output_dir;
  double eps_fraction=1e-4;
  double luminosity_fraction=2e-3;
  int stable_zones=8;
  bool self_test=false;
};

Options ParseOptions(int argc, char **argv) {
  Options options;
  if (argc==2 && std::string(argv[1])=="--self-test") { options.self_test=true; return options; }
  if (argc<2) throw std::runtime_error("usage: analyze_rsg_profile PROFILE --output-dir DIR [--eps-fraction X] [--luminosity-fraction X] [--stable-zones N]");
  options.profile=argv[1];
  for (int i=2;i<argc;++i) {
    const std::string arg=argv[i];
    if (i+1>=argc) throw std::runtime_error("missing value after "+arg);
    const std::string value=argv[++i];
    if (arg=="--output-dir") options.output_dir=value;
    else if (arg=="--eps-fraction") options.eps_fraction=std::stod(value);
    else if (arg=="--luminosity-fraction") options.luminosity_fraction=std::stod(value);
    else if (arg=="--stable-zones") options.stable_zones=std::stoi(value);
    else throw std::runtime_error("unknown option: "+arg);
  }
  if (options.output_dir.empty()) throw std::runtime_error("--output-dir is required");
  return options;
}

std::string CsvQuote(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (char c : value) {
    escaped.push_back(c);
    if (c == '"') escaped.push_back('"');
  }
  return '"' + escaped + '"';
}

int RunInventory(int argc, char **argv) {
  if (argc < 4) {
    throw std::runtime_error("usage: analyze_rsg_profile --inventory OUTPUT.csv ROOT [ROOT ...]");
  }
  const fs::path output_path = argv[2];
  std::vector<fs::path> profiles;
  for (int argument = 3; argument < argc; ++argument) {
    const fs::path root = argv[argument];
    if (fs::is_regular_file(root)) {
      profiles.push_back(root);
      continue;
    }
    std::error_code error;
    fs::recursive_directory_iterator iterator(
        root, fs::directory_options::skip_permission_denied, error);
    const fs::recursive_directory_iterator end;
    while (iterator != end) {
      if (error) { error.clear(); iterator.increment(error); continue; }
      if (iterator->is_regular_file(error)) {
        const std::string filename = iterator->path().filename().string();
        if (filename.rfind("profile", 0) == 0 && iterator->path().extension() == ".data") {
          profiles.push_back(iterator->path());
        }
      }
      iterator.increment(error);
    }
  }
  std::sort(profiles.begin(), profiles.end());
  profiles.erase(std::unique(profiles.begin(), profiles.end()), profiles.end());
  fs::create_directories(output_path.parent_path().empty() ? fs::path(".") : output_path.parent_path());
  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("cannot write inventory: " + output_path.string());
  out << "path,model_number,star_age_yr,initial_mass_Msun,star_mass_Msun,initial_z,Teff_K,photosphere_L_Lsun,photosphere_r_Rsun,he_core_mass_Msun,rsg_candidate\n";
  std::size_t valid = 0, candidates = 0, rejected = 0;
  for (const auto &path : profiles) {
    try {
      const auto header = ReadMesaHeader(path);
      const double model = HeaderNumber(header,"model_number");
      const double age = HeaderNumber(header,"star_age");
      const double initial_mass = HeaderNumber(header,"initial_mass");
      const double star_mass = HeaderNumber(header,"star_mass");
      const double z = HeaderNumber(header,"initial_z");
      const double teff = HeaderNumber(header,"Teff");
      const double luminosity = HeaderNumber(header,"photosphere_L");
      const double radius = HeaderNumber(header,"photosphere_r");
      const double he_core = HeaderNumber(header,"he_core_mass");
      const bool candidate = std::isfinite(star_mass) && std::isfinite(teff) &&
                             std::isfinite(radius) && std::isfinite(luminosity) &&
                             star_mass >= 8.0 && teff <= 5000.0 && radius >= 100.0 &&
                             luminosity >= 1000.0;
      out << CsvQuote(path.string());
      for (double value : {model,age,initial_mass,star_mass,z,teff,luminosity,radius,he_core}) {
        out << ',';
        if (std::isfinite(value)) out << std::setprecision(17) << value;
      }
      out << ',' << (candidate ? 1 : 0) << '\n';
      ++valid;
      if (candidate) ++candidates;
    } catch (const std::exception &) {
      ++rejected;
    }
  }
  std::cout << "inventory_profiles=" << profiles.size() << '\n'
            << "valid_headers=" << valid << '\n'
            << "rsg_candidates=" << candidates << '\n'
            << "rejected_headers=" << rejected << '\n'
            << "output=" << output_path << '\n';
  return 0;
}

int RunSensitivity(int argc, char **argv) {
  if (argc != 4) {
    throw std::runtime_error("usage: analyze_rsg_profile --sensitivity PROFILE OUTPUT.csv");
  }
  const fs::path profile_path = argv[2];
  const fs::path output_path = argv[3];
  const Profile profile = ReadMesaProfile(profile_path);
  const Boundary boundary = FindConvectiveBoundary(profile);
  fs::create_directories(output_path.parent_path().empty() ? fs::path(".") : output_path.parent_path());
  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("cannot write sensitivity table: " + output_path.string());
  out << "eps_nuc_over_peak_max,relative_L_minus_Lphot_max,stable_zones,core_cut_Msun,boundary_Msun,radiative_layer_Msun,fallback\n";
  const std::array<double,4> eps_values = {1e-3,1e-4,1e-5,1e-6};
  const std::array<double,4> luminosity_values = {1e-2,5e-3,2e-3,1e-3};
  for (double eps : eps_values) {
    for (double luminosity : luminosity_values) {
      const CoreCut core = FindCoreCut(profile,boundary.mass,eps,luminosity,8);
      out << std::scientific << std::setprecision(16)
          << eps << ',' << luminosity << ",8," << core.mass << ','
          << boundary.mass << ',' << boundary.mass-core.mass << ','
          << (core.fallback ? 1 : 0) << '\n';
    }
  }
  std::cout << "sensitivity_cases=16\noutput=" << output_path << '\n';
  return 0;
}

int RunReferenceGate(int argc, char **argv) {
  if (argc != 3) {
    throw std::runtime_error("usage: analyze_rsg_profile --reference-gate PROFILE");
  }
  const fs::path profile_path = argv[2];
  const std::string expected_sha =
      "b2f40ff36c3cd54bdd1fb19d4d6d4187db65db185928b3d259481170270c371c";
  const std::string sha = FileSha256(profile_path);
  Require(sha == expected_sha, "reference profile SHA-256");
  const Profile profile = ReadMesaProfile(profile_path);
  const Boundary boundary = FindConvectiveBoundary(profile);
  const CoreCut core = FindCoreCut(profile,boundary.mass,1e-4,2e-3,8);
  const auto close = [](double actual, double expected, double tolerance,
                        const std::string &name) {
    Require(std::abs(actual-expected) <= tolerance,
            name + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
  };
  Require(profile.zones == 1625, "reference zone count");
  close(HeaderNumber(profile,"star_mass",0.0),21.0,1e-12,"star mass");
  close(HeaderNumber(profile,"initial_z",0.0),0.02,1e-14,"metallicity");
  close(HeaderNumber(profile,"photosphere_r",0.0),895.4891516260694,1e-9,"photospheric radius");
  close(HeaderNumber(profile,"Teff",0.0),3563.8804369597915,1e-9,"effective temperature");
  close(core.mass,5.703236125105857,1e-12,"operational core cut");
  close(boundary.mass,6.296636507949257,1e-12,"radiative-convective boundary");
  Require(core.mass < boundary.mass, "retained radiative layer");
  std::cout << "reference gate passed\n"
            << "sha256=" << sha << '\n'
            << std::setprecision(12)
            << "core_cut_Msun=" << core.mass << '\n'
            << "boundary_Msun=" << boundary.mass << '\n'
            << "radiative_layer_Msun=" << boundary.mass-core.mass << '\n';
  return 0;
}

int Run(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "--inventory") {
    return RunInventory(argc, argv);
  }
  if (argc >= 2 && std::string(argv[1]) == "--sensitivity") {
    return RunSensitivity(argc, argv);
  }
  if (argc >= 2 && std::string(argv[1]) == "--reference-gate") {
    return RunReferenceGate(argc, argv);
  }
  const Options options=ParseOptions(argc,argv);
  if (options.self_test) { SelfTest(); return 0; }
  const Profile profile=ReadMesaProfile(options.profile);
  const Boundary boundary=FindConvectiveBoundary(profile);
  const CoreCut core=FindCoreCut(profile,boundary.mass,options.eps_fraction,
                                 options.luminosity_fraction,options.stable_zones);
  if (!(core.mass<boundary.mass)) throw std::runtime_error("selected core cut erases the radiative intershell");
  const Table table=BuildDerived(profile);
  fs::create_directories(options.output_dir);
  WriteCsv(options.output_dir/"mesa_rsg_layers.csv",table);
  WriteSvg(options.output_dir/"mesa_rsg_layers.svg",table,core.mass,boundary.mass);
  const auto &mass=table.at("mass_Msun");
  const auto &eps=table.at("eps_nuc_erg_g_s");
  double exterior_nuclear_luminosity=0.0;
  for (std::size_t i=0;i<mass.size();++i) {
    const double dm=(i==0?mass[i]:mass[i]-mass[i-1])*kMsun;
    if (mass[i]>=core.mass) exterior_nuclear_luminosity+=eps[i]*dm/kLsun;
  }
  const std::string sha=FileSha256(options.profile);
  WriteMetadata(options.output_dir/"mesa_rsg_metadata.json",options.profile,sha,
                profile,boundary,core,table,options.eps_fraction,
                options.luminosity_fraction,options.stable_zones,
                exterior_nuclear_luminosity);
  std::cout<<std::setprecision(12)
           <<"core_cut_Msun="<<core.mass<<'\n'
           <<"boundary_Msun="<<boundary.mass<<'\n'
           <<"radiative_layer_Msun="<<boundary.mass-core.mass<<'\n'
           <<"zones="<<profile.zones<<'\n'
           <<"source_sha256="<<sha<<'\n';
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try { return Run(argc,argv); }
  catch (const std::exception &error) {
    std::cerr<<"error: "<<error.what()<<'\n';
    return 1;
  }
}
