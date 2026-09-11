"""Regression test for stable FLD closure and transparent-face limits."""

import os
import subprocess

import scripts.utils.athena as athena


_PROBE = r'''
#include "fld/fld.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace {
bool Close(const long double a, const long double b) {
  const long double scale = std::max(std::abs(a), std::abs(b));
  return std::abs(a - b) <= 32.0L*std::numeric_limits<Real>::epsilon()
      * std::max(1.0L, scale);
}

bool Check(bool condition, const char *message) {
  if (!condition) std::cerr << "FAIL " << message << "\n";
  return condition;
}
}  // namespace

int main() {
  using namespace RadFLD;
  bool ok = true;
  const Real one_third = static_cast<Real>(ONE_3RD);
  const Real values[] = {
      0.0, std::numeric_limits<Real>::denorm_min(), 1.0,
      static_cast<Real>(1.0e155), std::numeric_limits<Real>::max()
  };
  for (const Real value : values) {
    const long double r = static_cast<long double>(value);
    const long double denominator = 6.0L + 3.0L*r + r*r;
    const long double reference = (2.0L + r)/denominator;
    const Real lambda = FluxLimiter(value, false);
    const Real lambda_r = FluxLimiterTimesR(value, false);
    const Real chi = EddingtonFactor(value, false);
    ok &= Check(std::isfinite(lambda) && lambda >= 0.0 && lambda <= one_third,
                "lambda range");
    ok &= Check(std::isfinite(lambda_r) && lambda_r >= 0.0 && lambda_r <= 1.0,
                "lambda*r range");
    ok &= Check(std::isfinite(chi) && chi >= one_third && chi <= 1.0,
                "chi range");
    if (value <= 1.0) {
      ok &= Check(Close(static_cast<long double>(lambda), reference),
                  "diffusion-side reference");
    } else {
      const long double reference_lambda_r =
          (1.0L + 2.0L/r)/(1.0L + 3.0L/r + 6.0L/(r*r));
      ok &= Check(Close(static_cast<long double>(lambda_r), reference_lambda_r),
                  "streaming-side reference");
    }
  }

  const Real infinity = std::numeric_limits<Real>::infinity();
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  ok &= Check(FluxLimiter(infinity, false) == 0.0, "infinite lambda");
  ok &= Check(FluxLimiterTimesR(infinity, false) == 1.0,
              "infinite lambda*r");
  ok &= Check(EddingtonFactor(infinity, false) == 1.0, "infinite chi");
  ok &= Check(std::isnan(FluxLimiter(nan, false)), "NaN ratio rejection");

  const ClosureValues transparent = EvaluateClosure(1.0, 0.0, 1.0, false);
  ok &= Check(transparent.valid && transparent.lambda == 0.0
              && transparent.lambda_r == 1.0 && transparent.chi == 1.0
              && transparent.lambda_over_opacity == 1.0,
              "zero-opacity streaming limit");
  const ClosureValues vacuum = EvaluateClosure(1.0, 0.0, 0.0, false);
  ok &= Check(vacuum.valid && vacuum.lambda_over_opacity == 0.0,
              "zero-opacity zero-energy limit");
  const ClosureValues zero = EvaluateClosure(0.0, 0.0, 0.0, false);
  ok &= Check(zero.valid && zero.lambda == one_third
              && zero.lambda_over_opacity == 0.0, "zero-gradient limit");
  ok &= Check(!EvaluateClosure(1.0, -1.0, 1.0, false).valid,
              "negative opacity rejection");
  ok &= Check(!EvaluateClosure(1.0, 1.0, -1.0, false).valid,
              "negative Er rejection");
  ok &= Check(!EvaluateClosure(1.0, 1.0, nan, false).valid,
              "non-finite Er rejection");
  ok &= Check(!EvaluateClosure(1.0, 0.0, 1.0, true).valid,
              "fixed transparent face rejection");
  ok &= Check(FaceOpacity(0.0, 0.0, 1.0) == 0.0,
              "zero face opacity");
  ok &= Check(std::isfinite(FaceOpacity(std::numeric_limits<Real>::max(),
                                        std::numeric_limits<Real>::max(), 1.0)),
              "maximum face opacity");
  ok &= Check(ok, "all closure checks");
  if (ok) std::cout << "PASS\n";
  return ok ? 0 : 1;
}
'''


def _configure_and_compile(name, precision):
    arguments = ['nrmgfld']
    if precision == 'single':
        arguments.append('float')
    athena.configure(*arguments, prob='nrfld_sound', coord='cartesian',
                     eos='adiabatic', cxx='g++')
    os.makedirs('bin', exist_ok=True)
    source_root = os.path.abspath(athena.athena_rel_path)
    output = os.path.abspath(os.path.join('bin', name))
    command = ['g++', '-std=c++11', '-O2', '-I' + os.path.join(source_root, 'src'),
               '-x', 'c++', '-',
               '-o', output]
    subprocess.run(command, input=_PROBE.encode(), check=True)


def prepare(**kwargs):
    _configure_and_compile('stable_fld_closure_double', 'double')
    _configure_and_compile('stable_fld_closure_single', 'single')


def run(**kwargs):
    for precision in ('double', 'single'):
        output = os.path.join('bin', 'stable_fld_closure_{}.log'.format(precision))
        with open(output, 'w') as log:
            subprocess.check_call(
                ['./stable_fld_closure_{}'.format(precision)],
                cwd='bin', stdout=log, stderr=subprocess.STDOUT)


def analyze():
    for precision in ('double', 'single'):
        output = os.path.join('bin', 'stable_fld_closure_{}.log'.format(precision))
        with open(output, 'r') as log:
            if log.read().strip() != 'PASS':
                return False
    return True
