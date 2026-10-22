#ifndef UTILS_PERFORMANCE_TIMER_HPP_
#define UTILS_PERFORMANCE_TIMER_HPP_
// Optional, main-thread wall timing. No communication inside the measured window.
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "../athena.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class PerformanceTimer {
 public:
  enum Part {other, explicit_integrator, gravity_solve, fld_prepare, fld_solve,
             fld_sync, fld_finish, amr, output, count};
  explicit PerformanceTimer(ParameterInput *pin)
      : enabled_(pin->DoesParameterExist("performance", "enabled") &&
                 pin->GetBoolean("performance", "enabled")),
        warmup_(enabled_ ? pin->GetOrAddInteger("performance", "warmup", 5) : 0) {
    if (warmup_ < 0) {
      std::stringstream msg;
      msg << "performance/warmup must be nonnegative";
      ATHENA_ERROR(msg);
    }
  }

  static double Now() {
#ifdef MPI_PARALLEL
    return MPI_Wtime();
#else
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
  }
  void BeginStep(int cycle, double dt, double cells) {
    if (!enabled_ || seen_++ < warmup_) return;
    if (steps_.empty()) {
#ifdef MPI_PARALLEL
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      start_ = Now();
    }
    active_ = true;
    current_ = other;
    last_ = Now();
    steps_.push_back({cycle, dt, cells, -1, 0.0, 0.0, "not_run"});
  }
  void Mark(Part next) {
    if (!active_) return;
    const double now = Now();
    seconds_[current_] += now - last_;
    last_ = now;
    current_ = next;
  }
  void Newton(int iterations, double l2, double maxnorm, const char *reason) {
    if (!active_) return;
    auto &s = steps_.back();
    s.iterations = iterations;
    s.l2 = l2;
    s.maxnorm = maxnorm;
    s.reason = reason;
  }
  void EndStep() {
    if (!active_) return;
    Mark(other);
    end_ = last_;
    active_ = false;
  }
  void Write() const {
    if (!enabled_) return;
    const char *names[] = {"other", "explicit_integrator", "gravity_solve",
        "fld_prepare", "fld_solve", "fld_sync", "fld_finish", "amr", "output"};
    std::ofstream out("performance_rank" + std::to_string(Globals::my_rank) + ".json");
    if (!out) {
      std::stringstream msg;
      msg << "Cannot write performance timing file";
      ATHENA_ERROR(msg);
    }
    out << std::setprecision(17) << "{\"schema_version\":1,\"rank\":"
        << Globals::my_rank << ",\"warmup\":" << warmup_
        << ",\"wall_seconds\":" << (steps_.empty() ? 0.0 : end_ - start_)
        << ",\"seconds\":{";
    for (int i = 0; i < count; ++i)
      out << (i ? "," : "") << '"' << names[i] << "\":" << seconds_[i];
    out << "},\"steps\":[";
    for (std::size_t i = 0; i < steps_.size(); ++i) {
      const auto &s = steps_[i];
      out << (i ? "," : "") << "{\"cycle\":" << s.cycle << ",\"dt\":" << s.dt
          << ",\"global_cells\":" << s.cells << ",\"newton_iterations\":"
          << s.iterations << ",\"newton_l2\":" << s.l2
          << ",\"newton_max\":" << s.maxnorm << ",\"newton_reason\":\""
          << s.reason << "\"}";
    }
    out << "]}\n";
  }
 private:
  struct Step {
    int cycle;
    double dt, cells;
    int iterations;
    double l2, maxnorm;
    std::string reason;
  };
  bool enabled_, active_ = false;
  int warmup_, seen_ = 0;
  Part current_ = other;
  double last_ = 0.0, start_ = 0.0, end_ = 0.0;
  std::array<double, count> seconds_{};
  std::vector<Step> steps_;
};
#endif
