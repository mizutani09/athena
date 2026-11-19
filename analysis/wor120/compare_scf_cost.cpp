#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Cost {
  double final_time;
  long long cycles;
  long long meshblocks;
  double zone_cycles;
  double cpu_seconds;
  double zone_cycles_per_cpu_second;
  double wall_seconds;
  double launcher_maxrss_kib;
  double slurm_elapsed_seconds;
  double slurm_average_rss_kib;
  double slurm_maximum_rss_kib;
};

std::string Trim(const std::string& text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last-first+1);
}

double ParseNumber(const std::string& text, const std::string& label) {
  std::stringstream input(Trim(text));
  double value = 0.0;
  input >> value;
  if (!input || !std::isfinite(value)) {
    throw std::runtime_error("cannot parse " + label + ": " + text);
  }
  return value;
}

long long ParseInteger(const std::string& text, const std::string& label) {
  std::stringstream input(Trim(text));
  long long value = 0;
  input >> value;
  if (!input) throw std::runtime_error("cannot parse " + label + ": " + text);
  return value;
}

double ParseMemoryKib(const std::string& text) {
  const std::string value = Trim(text);
  if (value.empty()) throw std::runtime_error("empty Slurm memory field");
  const char suffix = value[value.size()-1];
  double multiplier = 1.0;
  std::string number = value;
  if (suffix == 'K' || suffix == 'M' || suffix == 'G' || suffix == 'T') {
    number.erase(number.size()-1);
    if (suffix == 'M') multiplier = 1024.0;
    if (suffix == 'G') multiplier = 1024.0*1024.0;
    if (suffix == 'T') multiplier = 1024.0*1024.0*1024.0;
  }
  return ParseNumber(number, "Slurm memory")*multiplier;
}

std::vector<std::string> Split(const std::string& line, char delimiter) {
  std::vector<std::string> fields;
  std::stringstream input(line);
  std::string field;
  while (std::getline(input, field, delimiter)) fields.push_back(field);
  return fields;
}

void ReadAthenaStdout(const std::string& filename, Cost* cost) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  bool have_final = false;
  bool have_blocks = false;
  bool have_zone_cycles = false;
  bool have_cpu = false;
  bool have_rate = false;
  std::string line;
  while (std::getline(input, line)) {
    if (line.compare(0, 5, "time=") == 0 && line.find(" cycle=") != std::string::npos) {
      const std::size_t cycle_pos = line.find(" cycle=");
      cost->final_time = ParseNumber(line.substr(5, cycle_pos-5), "final time");
      cost->cycles = ParseInteger(line.substr(cycle_pos+7), "cycle count");
      have_final = true;
    } else if (line.compare(0, 22, "Number of MeshBlocks =") == 0) {
      const std::size_t equal = line.find('=');
      cost->meshblocks = ParseInteger(line.substr(equal+1), "MeshBlock count");
      have_blocks = true;
    } else if (line.compare(0, 13, "zone-cycles =") == 0) {
      cost->zone_cycles = ParseNumber(line.substr(13), "zone cycles");
      have_zone_cycles = true;
    } else if (line.compare(0, 16, "cpu time used  =") == 0) {
      cost->cpu_seconds = ParseNumber(line.substr(16), "CPU seconds");
      have_cpu = true;
    } else if (line.compare(0, 24, "zone-cycles/cpu_second =") == 0) {
      cost->zone_cycles_per_cpu_second = ParseNumber(
          line.substr(24), "zone cycles per CPU second");
      have_rate = true;
    }
  }
  if (!have_final || !have_blocks || !have_zone_cycles || !have_cpu || !have_rate) {
    throw std::runtime_error("incomplete Athena timing footer: " + filename);
  }
}

void ReadTimeLog(const std::string& filename, Cost* cost) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  std::string key;
  bool have_wall = false;
  bool have_rss = false;
  while (input >> key) {
    const std::size_t equal = key.find('=');
    if (equal == std::string::npos) continue;
    const std::string name = key.substr(0, equal);
    const double value = ParseNumber(key.substr(equal+1), name);
    if (name == "wall_s") {
      cost->wall_seconds = value;
      have_wall = true;
    } else if (name == "maxrss_kb") {
      cost->launcher_maxrss_kib = value;
      have_rss = true;
    }
  }
  if (!have_wall || !have_rss) {
    throw std::runtime_error("incomplete /usr/bin/time log: " + filename);
  }
}

void ReadSacct(const std::string& filename, Cost* cost) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  std::string line;
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = Split(line, '|');
    if (fields.size() < 7 || Trim(fields[0]).empty()) continue;
    if (Trim(fields[1]) != "COMPLETED") {
      throw std::runtime_error("MPI step did not complete: " + Trim(fields[1]));
    }
    cost->slurm_elapsed_seconds = ParseNumber(fields[2], "Slurm elapsed seconds");
    cost->slurm_average_rss_kib = ParseMemoryKib(fields[3]);
    cost->slurm_maximum_rss_kib = ParseMemoryKib(fields[4]);
    return;
  }
  throw std::runtime_error("no Slurm step accounting row: " + filename);
}

Cost ReadCost(const std::string& stdout_file, const std::string& time_file,
              const std::string& sacct_file) {
  Cost cost = {};
  ReadAthenaStdout(stdout_file, &cost);
  ReadTimeLog(time_file, &cost);
  ReadSacct(sacct_file, &cost);
  return cost;
}

double Ratio(double numerator, double denominator, const char* label) {
  if (!(denominator > std::numeric_limits<double>::min())) {
    throw std::runtime_error(std::string("non-positive denominator for ") + label);
  }
  return numerator/denominator;
}

void WriteMethod(std::ostream& output, const char* name, const Cost& cost,
                 bool trailing_comma) {
  output << "  \"" << name << "\": {\n"
         << "    \"final_time_code\": " << cost.final_time << ",\n"
         << "    \"cycles\": " << cost.cycles << ",\n"
         << "    \"meshblocks\": " << cost.meshblocks << ",\n"
         << "    \"zone_cycles\": " << cost.zone_cycles << ",\n"
         << "    \"athena_cpu_seconds\": " << cost.cpu_seconds << ",\n"
         << "    \"wall_seconds\": " << cost.wall_seconds << ",\n"
         << "    \"slurm_step_elapsed_seconds\": "
         << cost.slurm_elapsed_seconds << ",\n"
         << "    \"zone_cycles_per_cpu_second\": "
         << cost.zone_cycles_per_cpu_second << ",\n"
         << "    \"slurm_average_rss_kib_per_rank\": "
         << cost.slurm_average_rss_kib << ",\n"
         << "    \"slurm_maximum_rss_kib_per_rank\": "
         << cost.slurm_maximum_rss_kib << ",\n"
         << "    \"launcher_maxrss_kib\": " << cost.launcher_maxrss_kib << "\n"
         << "  }" << (trailing_comma ? "," : "") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 8) {
      std::cerr << "usage: " << argv[0]
                << " X2_STDOUT X2_TIME X2_SACCT 4WZ_STDOUT 4WZ_TIME"
                << " 4WZ_SACCT OUTPUT.json\n";
      return 2;
    }
    const Cost x2 = ReadCost(argv[1], argv[2], argv[3]);
    const Cost wz = ReadCost(argv[4], argv[5], argv[6]);
    if (x2.meshblocks != wz.meshblocks) {
      throw std::runtime_error("A/B MeshBlock counts differ");
    }
    const double final_time_scale = std::max(std::abs(x2.final_time),
                                              std::abs(wz.final_time));
    if (final_time_scale == 0.0
        || std::abs(x2.final_time-wz.final_time)/final_time_scale > 1.0e-3) {
      throw std::runtime_error("A/B final times differ by more than 0.1 percent");
    }

    std::ofstream output(argv[7]);
    if (!output) throw std::runtime_error("cannot create output JSON");
    output << std::scientific << std::setprecision(12) << "{\n";
    WriteMethod(output, "rk2_x2", x2, true);
    WriteMethod(output, "rk4_4wz", wz, true);
    output << "  \"rk4_4wz_over_rk2_x2\": {\n"
           << "    \"cycle_count_ratio\": "
           << Ratio(static_cast<double>(wz.cycles), static_cast<double>(x2.cycles),
                    "cycle count") << ",\n"
           << "    \"athena_cpu_time_ratio\": "
           << Ratio(wz.cpu_seconds, x2.cpu_seconds, "CPU time") << ",\n"
           << "    \"wall_time_ratio\": "
           << Ratio(wz.wall_seconds, x2.wall_seconds, "wall time") << ",\n"
           << "    \"cost_per_zone_cycle_ratio\": "
           << Ratio(wz.cpu_seconds/wz.zone_cycles,
                    x2.cpu_seconds/x2.zone_cycles, "cost per zone cycle") << ",\n"
           << "    \"throughput_ratio\": "
           << Ratio(wz.zone_cycles_per_cpu_second,
                    x2.zone_cycles_per_cpu_second, "throughput") << ",\n"
           << "    \"average_rss_ratio\": "
           << Ratio(wz.slurm_average_rss_kib, x2.slurm_average_rss_kib,
                    "average RSS") << ",\n"
           << "    \"maximum_rss_ratio\": "
           << Ratio(wz.slurm_maximum_rss_kib, x2.slurm_maximum_rss_kib,
                    "maximum RSS") << "\n"
           << "  }\n"
           << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "compare_scf_cost: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
