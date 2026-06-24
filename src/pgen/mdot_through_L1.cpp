//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mdot_through_L1.cpp
//! \brief Problem generator for mass transfer through L1 point in a binary system.
//! REFERENCE:
//========================================================================================

// C headers

// C/C++ headers
#include <cmath>      // sqrt(), sin(), cos()
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


namespace {
  // general
  Real GUnit, massUnit, lengthUnit;
  Real rhoUnit, egasUnit, velUnit, TimeUnit, TUnit;
  Real a_r_unit, a_r_sim;

  // paramteters for boundary
  Real vel_lim;

  // int rk_cycle;

  Real Gcnst = 6.67259e-8; // gravitational constant [cm3/g/s2]
  Real Rgas  = 8.3e7; // erg * K^-1 * mol^-1
  Real Msun  = 1.99e33; // Solar mass in g
  Real Rsun  = 6.96e10; // Solar radius in cm
  Real mu_mmw; // mean molecular weight
  Real a_r_dim = 7.5657e-15; // radiation constant in cgs units

//   // parameters for system
  Real separation_dim, separation_sim = 1.0; // Fixed.
  Real Mtot_dim, Mtot_sim = 1.0; // Fixed.
  Real Omega_orbit;
//   Real Tstar_dim, Tstar_sim;
  Real mass_ratio;
  Real grav_eps;
  bool add_centrifugal, add_coriolis;
  Real Mdot_dim, Mdot_sim;
  Real R_sink;

  
  Real x_Ls[3], phi_Ls[3];


  Real GMtot_dim, GMtot_sim;
//   std::string GRAVITY_PROFILE;

    // parameters for ruser_meshblock_data
  int UGRAVD    = 0;
  int UGRAVA    = 1;
  int UCENTRI   = 2;
  int UCORIOL   = 3;

  //   // parameters for iuser_meshblock_data
  // int TSTEP_COUNTER = 0;

    // parameters for user_out_var
  int iuov_max;
} //namespace


void UserBNDInnerX1(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh);
void UserBNDOuterX1(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh);
void UserBNDInnerX2(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh);
void UserBNDOuterX2(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh);
void UserBNDInnerX3(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh);
  void UserBNDOuterX3(MeshBlock *pmb, Coordinates *pco,
    AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
    int is, int ie, int js, int je, int ks, int ke, int ngh);
void SourceStar(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                AthenaArray<Real> &cons_scalar);
Real CalcDensity(Real r);
Real CalcTemperature(Real r);
Real CalcRochePotentialPre(Real x1);
void FindLocalMaximum(Real x_start, Real x_end,
                      Real &x_max, Real &phi_max);
void TestForRoche(Real (&x_Ls)[3], Real (&phi_Ls)[3]);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // check configuration
  if (COORDINATE_SYSTEM != "cartesian") {
    // raise error
    std::stringstream msg;
    msg << "### FATAL ERROR in mdot_through_L1.cpp InitUserMeshData" << std::endl
        << "This pgen only for cartesian coordinate." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mdot_through_L1.cpp InitUserMeshData" << std::endl
        << "Magnetic field is not supported now." << std::endl;
    ATHENA_ERROR(msg);
  }

  // initialize units
  Mtot_dim = pin->GetReal("problem", "Mtot_dim") * Msun;
  separation_dim = pin->GetReal("problem", "separation_dim") * Rsun;
  massUnit = Mtot_dim;
  lengthUnit = separation_dim;
  GUnit = Gcnst;
  rhoUnit = Mtot_dim / std::pow(separation_dim, 3);
  egasUnit = GUnit*std::pow(massUnit, 2)/std::pow(lengthUnit, 4);
  velUnit = std::sqrt(GUnit*massUnit/lengthUnit);
  TimeUnit = lengthUnit / velUnit;
  mu_mmw = pin->GetReal("hydro", "mu_mmw");
  TUnit = Gcnst * Mtot_dim / separation_dim / Rgas * mu_mmw;
  a_r_unit = std::pow(Rgas/mu_mmw, 4)*massUnit/std::pow(GUnit*massUnit, 3);
  a_r_sim = a_r_dim/a_r_unit;

  // // initialize parameters for main star
  // M = Mstar_dim / MUnit;
  GMtot_sim = Gcnst * Mtot_dim / (GUnit * massUnit);
  // R = Rstar_dim / RUnit;
  Omega_orbit = 1.0; // std::sqrt(G*Mtot_sim/std::pow(separation_sim, 3));

  // print units
  if (Globals::my_rank == 0) {
    std::cout << "### INFO in mdot_through_L1.cpp InitUserMeshData" << std::endl;
    std::cout << "Units: Mtot_dim=" << Mtot_dim << " g, separation_dim=" << separation_dim << " cm, GUnit=" << GUnit << " cm^3/g/s^2\n";
    std::cout << "       rhoUnit=" << rhoUnit << " g/cm^3, egasUnit=" << egasUnit << " erg/cm^3, velUnit=" << velUnit << " cm/s\n";
    std::cout << "       TimeUnit=" << TimeUnit << " s, TUnit=" << TUnit << " K, a_r_unit=" << a_r_unit << " erg/cm^3\n";
    std::cout << "       a_r_sim=" << a_r_sim << " in code units\n";
  }
  
  // initialize parameters for system
  mass_ratio = pin->GetReal("problem", "mass_ratio");
  grav_eps = pin->GetReal("problem", "grav_eps");
  add_centrifugal = pin->GetOrAddBoolean("problem", "centrifugal", true);
  add_coriolis = pin->GetOrAddBoolean("problem", "coriolis", true);
  Mdot_dim = pin->GetOrAddReal("problem", "Mdot_dim", -1.0);
  Mdot_sim = Mdot_dim / (massUnit / TimeUnit);
  if (Mdot_dim < 0.0) {
    if (Globals::my_rank == 0)
      std::cout << "### WARNING in mdot_through_L1.cpp InitUserMeshData" << std::endl
                << "Mdot_dim is not specified in the input file. Do not add mass accretion." << std::endl;
  }
  R_sink = pin->GetOrAddReal("problem", "R_sink", -1.0);
  if (R_sink < 0.0) {
    if (Globals::my_rank == 0)
      std::cout << "### WARNING in mdot_through_L1.cpp InitUserMeshData" << std::endl
                << "R_sink is not specified in the input file. Do not add sink accretion." << std::endl;
  }
  
  // for the outer region and boundary condition
  // Real r_min = std::min(std::abs(mesh_size.x3max), std::abs(mesh_size.x3min));
  // vel_lim = CalVelLim(r_min, semi_major_init, grav_ratio, omega_orbit);
  // Real Tcorona_dim = pin->GetReal("problem","Tcorona_dim");
  // tem_out = Tcorona_dim / TUnit;

  // user-defined boundary conditions
  if (pin->GetString("mesh", "ix1_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::inner_x1, UserBNDInnerX1);
  if (pin->GetString("mesh", "ox1_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::outer_x1, UserBNDOuterX1);
  if (pin->GetString("mesh", "ix2_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::inner_x2, UserBNDInnerX2);
  if (pin->GetString("mesh", "ox2_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::outer_x2, UserBNDOuterX2);
  if (pin->GetString("mesh", "ix3_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::inner_x3, UserBNDInnerX3);
  if (pin->GetString("mesh", "ox3_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::outer_x3, UserBNDOuterX3);

  EnrollUserExplicitSourceFunction(SourceStar);
}


//======================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//  \brief Initialize user-defined mesh block data
//======================================================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  int rdata_size = 0;
  rdata_size += 2; // for gravitational acceleration of donor and accretor
  rdata_size += 1; // for centrifugal force
  rdata_size += 1; // for coriolis force
  AllocateRealUserMeshBlockDataField(rdata_size);

  // gravitational accelerations in all the directions
  ruser_meshblock_data[UGRAVD].NewAthenaArray(3,ncells3,ncells2,ncells1);
  ruser_meshblock_data[UGRAVA].NewAthenaArray(3,ncells3,ncells2,ncells1);

  ruser_meshblock_data[UCENTRI].NewAthenaArray(3,ncells3,ncells2,ncells1);
  ruser_meshblock_data[UCORIOL].NewAthenaArray(3,ncells3,ncells2,ncells1);


  // int idata_size = 0;
  // idata_size += 1; // for test counter
  // AllocateIntUserMeshBlockDataField(idata_size);

  // iuser_meshblock_data[TSTEP_COUNTER].NewAthenaArray(1);
  // iuser_meshblock_data[TSTEP_COUNTER](0) = 0;

  
  // user output variables
  int iuov = 0; // initialize
  iuov_max = 0;
  iuov_max += 1; // for ent
  iuov_max += 1; // for sound speed
  iuov_max += 2; // for gravitational acceleration
  iuov_max += 2; // for centrifugal and coriolis forces
  iuov_max += 1; // for Mach

  AllocateUserOutputVariables(iuov_max);
  SetUserOutputVariableName(iuov, "ent"), iuov++;
  SetUserOutputVariableName(iuov, "sound"), iuov++;
  SetUserOutputVariableName(iuov, "gacc1"), iuov++;
  SetUserOutputVariableName(iuov, "gacc2"), iuov++;
  SetUserOutputVariableName(iuov, "centrifugal"), iuov++;
  SetUserOutputVariableName(iuov, "coriolis"), iuov++;
  SetUserOutputVariableName(iuov, "Mach"), iuov++;

  TestForRoche(x_Ls, phi_Ls);
  if (gid == 0) {
    std::cout << "Lagrange points: L1, L2, L3 = " << x_Ls[0] << ", " << x_Ls[1] << ", " << x_Ls[2] << std::endl;
    std::cout << "Potential at Lagrange points: L1, L2, L3 = " << phi_Ls[0] << ", " << phi_Ls[1] << ", " << phi_Ls[2] << std::endl;

    // info about sink
    Real dx = pcoord->dx1v(0);
    if (R_sink > 0.0) {
      std::cout << "R_sink/dx = " << R_sink/dx << std::endl;
    }
  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for polytropic sphere.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma_ad = 5.0/3.0;
  Real igm1_ad = 1.0 / (gamma_ad - 1.0);

  int il = is;//- NGHOST;
  int iu = ie;//+ NGHOST;
  int jl = js;//- NGHOST;
  int ju = je;//+ NGHOST;
  int kl = ks;//- NGHOST;
  int ku = ke;//+ NGHOST;

  for (int k=kl; k<=ku; ++k) {
    Real z = pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real y = pcoord->x2v(j);
      for (int i=il; i<=iu; ++i) {
        Real x = pcoord->x1v(i);
        Real r = std::sqrt(x*x + y*y + z*z);
        Real rho_res = CalcDensity(r);
        Real temp_res = CalcTemperature(r);
        Real pr_res = rho_res * temp_res;
        Real vx = 0.0, vy = 0.0, vz = 0.0;
        phydro->u(IDN,k,j,i) = rho_res;
        phydro->u(IM1,k,j,i) = rho_res * vx;
        phydro->u(IM2,k,j,i) = rho_res * vy;
        phydro->u(IM3,k,j,i) = rho_res * vz;

        if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN,k,j,i) = igm1_ad*pr_res;
        }
        
        // for the accelerations
        Real dx, dy, dz;

        // gravitational acceleration of donor
        dx = x + separation_sim;
        dy = y;
        dz = z;
        Real r_d = std::sqrt(SQR(dx) + SQR(dy) + SQR(dz) + SQR(grav_eps));
        Real grav_donor_pre = -GMtot_sim * (1.0/(1.0 + mass_ratio)) / CUBE(r_d);
        ruser_meshblock_data[UGRAVD](0,k,j,i) = grav_donor_pre*dx;
        ruser_meshblock_data[UGRAVD](1,k,j,i) = grav_donor_pre*dy;
        ruser_meshblock_data[UGRAVD](2,k,j,i) = grav_donor_pre*dz;
        
        // gravitational acceleration of accretor
        dx = x;
        dy = y;
        dz = z;
        Real r_a = std::sqrt(SQR(dx) + SQR(dy) + SQR(dz) + SQR(grav_eps));
        Real grav_accretor = -GMtot_sim * (1.0/(1.0 + mass_ratio)) / CUBE(r_a);
        ruser_meshblock_data[UGRAVA](0,k,j,i) = grav_accretor*dx;
        ruser_meshblock_data[UGRAVA](1,k,j,i) = grav_accretor*dy;
        ruser_meshblock_data[UGRAVA](2,k,j,i) = grav_accretor*dz;

        // centrifugal force
        if (add_centrifugal) {
          Real x_CM = (-separation_sim + mass_ratio*0.0) / (1.0 + mass_ratio);
          dx = x - x_CM;
          dy = y;
          dz = z;
          ruser_meshblock_data[UCENTRI](0,k,j,i) = SQR(Omega_orbit) * dx;
          ruser_meshblock_data[UCENTRI](1,k,j,i) = SQR(Omega_orbit) * dy;
          ruser_meshblock_data[UCENTRI](2,k,j,i) = 0.0;
        }

        // coriolis force
        if (add_coriolis) {
          ruser_meshblock_data[UCORIOL](0,k,j,i) =  2.0 * Omega_orbit * vy;
          ruser_meshblock_data[UCORIOL](1,k,j,i) = -2.0 * Omega_orbit * vx;
          ruser_meshblock_data[UCORIOL](2,k,j,i) =  0.0;
        }
      }
    }
  }
}


//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//! \brief User-defined work function for every time step.
//======================================================================================
void MeshBlock::UserWorkInLoop(void) {
  // for mass injection through L1 point
  if (Mdot_dim > 0.0) {
    Real gamma = 5.0/3.0;
    Real k_B = 1.380649e-16; // Boltzmann constant in cgs units
    Real m_H = 1.6735575e-24; // mass of hydrogen
    Real T_inj = 1.0e4; // injection temperature in K
    Real v_inj = 0.1; // injection velocity in code units
    Real H_L1 = 0.05; // scale height of the injection region in code units
    Real w_x = 0.1; // width of the injection region in x-direction in code units
    Real R_cut = 0.1; // cutoff radius of the injection region in y-z plane in code units
    Real t_delay = 1.0; // delay time for ramping up the mass injection rate in code units
    Real nozzle_norm = std::pow(2.0*M_PI, 1.5) * std::pow(H_L1, 2) * w_x;

    Real Mdot = Mdot_sim;
    Real ramp = 1.0;
    if (pmy_mesh->time < t_delay) {
      ramp = 0.5 * (1.0 - std::cos(PI * pmy_mesh->time / t_delay));
    }
    Real Mdot_now = ramp * Mdot;

    Real cs_iso2 = k_B * T_inj / (mu_mmw * m_H) / (velUnit * velUnit);
    Real u_inj = cs_iso2 / (gamma - 1.0);

    for (int k=ks; k<=ke; ++k) {
      Real z = pcoord->x3v(k);

      for (int j=js; j<=je; ++j) {
        Real y =pcoord->x2v(j);

        for (int i=is; i<=ie; ++i) {
          Real x = pcoord->x1v(i);

          Real xi = x - x_Ls[0];
          Real s2 = y*y + z*z;

          if (std::abs(xi) < 0.5*w_x && s2 < R_cut*R_cut) {
            Real W = std::exp(-0.5*s2/(H_L1*H_L1));

            Real Srho = Mdot_now * W / nozzle_norm;
            Real drho = Srho * pmy_mesh->dt;

            Real vx_inj = v_inj;
            Real vy_inj = 0.0;
            Real vz_inj = 0.0;

            phydro->w(IDN,k,j,i) += drho;
            phydro->w(IVX,k,j,i) = vx_inj;
            phydro->w(IVY,k,j,i) = vy_inj;
            phydro->w(IVZ,k,j,i) = vz_inj;
            phydro->w(IPR,k,j,i) = drho * u_inj * (gamma - 1.0);

            phydro->u(IDN,k,j,i) += drho;
            phydro->u(IM1,k,j,i) += drho * vx_inj;
            phydro->u(IM2,k,j,i) += drho * vy_inj;
            phydro->u(IM3,k,j,i) += drho * vz_inj;
            Real e_inj =
                u_inj
              + 0.5*(vx_inj*vx_inj + vy_inj*vy_inj + vz_inj*vz_inj);

            phydro->u(IEN,k,j,i) += drho * e_inj;
          }
        }
      }
    }
  }

  // for sink accretion
  if (R_sink > 0.0) {
    Real gamma = 5.0/3.0;
    Real rho_floor = 1.0e-10;
    Real p_floor = rho_floor * std::pow(10.0, 4);
    Real t_sink = 0.1 * pmy_mesh->dt;
    Real dR_sink = 0.1 * R_sink;

    Real dM_acc_local = 0.0;
    Real dPx_acc_local = 0.0;
    Real dPy_acc_local = 0.0;
    Real dPz_acc_local = 0.0;
    Real dJz_acc_local = 0.0;

    for (int k=ks; k <=ke; ++k) {
      Real z  = pcoord->x3v(k);
      Real dz = pcoord->dx3f(k);

      for (int j=js; j<=je; ++j) {
        Real y  = pcoord->x2v(j);
        Real dy = pcoord->dx2f(j);

        for (int i=is; i<=ie; ++i) {
          Real x  = pcoord->x1v(i);
          Real dx = pcoord->dx1f(i);

          Real r = std::sqrt(x*x + y*y + z*z);

          // Smooth sink weight
          Real W = 0.5 * (1.0 - std::tanh((r - R_sink) / dR_sink));

          // If far outside sink, skip
          if (W < 1.0e-8) continue;

          Real rho_old = phydro->u(IDN,k,j,i);
          Real mx_old  = phydro->u(IM1,k,j,i);
          Real my_old  = phydro->u(IM2,k,j,i);
          Real mz_old  = phydro->u(IM3,k,j,i);
          Real E_old   = phydro->u(IEN,k,j,i);

          // Exact exponential relaxation factor
          Real f = 1.0 - std::exp(-W * pmy_mesh->dt / t_sink);

          // Target state
          Real rho_targ = rho_floor;
          Real vx_targ  = 0.0;
          Real vy_targ  = 0.0;
          Real vz_targ  = 0.0;

          Real mx_targ = rho_targ * vx_targ;
          Real my_targ = rho_targ * vy_targ;
          Real mz_targ = rho_targ * vz_targ;

          Real E_targ =
              p_floor / (gamma - 1.0)
            + 0.5 * rho_targ
            * (vx_targ*vx_targ + vy_targ*vy_targ + vz_targ*vz_targ);

          // Relax conserved variables
          Real rho_new = rho_old + f * (rho_targ - rho_old);
          Real mx_new  = mx_old  + f * (mx_targ  - mx_old);
          Real my_new  = my_old  + f * (my_targ  - my_old);
          Real mz_new  = mz_old  + f * (mz_targ  - mz_old);
          Real E_new   = E_old   + f * (E_targ   - E_old);

          // Diagnostics: removed mass and momentum
          Real dV = dx * dy * dz;

          Real dm  = std::max(rho_old - rho_new, 0.0) * dV;
          Real dpx = (mx_old - mx_new) * dV;
          Real dpy = (my_old - my_new) * dV;
          Real dpz = (mz_old - mz_new) * dV;

          dM_acc_local  += dm;
          dPx_acc_local += dpx;
          dPy_acc_local += dpy;
          dPz_acc_local += dpz;

          // Angular momentum around accretor in rotating-frame velocity.
          // For inertial-frame angular momentum, add Omega x r to velocity/momentum.
          Real dJz = x * dpy - y * dpx;
          dJz_acc_local += dJz;

          // Apply update
          phydro->w(IDN,k,j,i) = rho_new;
          phydro->w(IVX,k,j,i) = mx_new / rho_new;
          phydro->w(IVY,k,j,i) = my_new / rho_new;
          phydro->w(IVZ,k,j,i) = mz_new / rho_new;
          phydro->w(IPR,k,j,i) = (E_new - 0.5*(mx_new*mx_new + my_new*my_new + mz_new*mz_new)/rho_new) * (gamma - 1.0);
          phydro->u(IDN,k,j,i) = rho_new;
          phydro->u(IM1,k,j,i) = mx_new;
          phydro->u(IM2,k,j,i) = my_new;
          phydro->u(IM3,k,j,i) = mz_new;
          phydro->u(IEN,k,j,i) = E_new;
        }
      }
    }
  }
}


//======================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//======================================================================================
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  Real gamma_ad = 5.0/3.0;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        Real vx = phydro->w(IVX,k,j,i);
        Real vy = phydro->w(IVY,k,j,i);
        Real vz = phydro->w(IVZ,k,j,i);
        Real dens = phydro->w(IDN,k,j,i);
        Real idens = 1.0 / dens;
        Real pres = phydro->w(IPR,k,j,i);

        int iuov = 0; // initialize
        // for ent
        user_out_var(iuov,k,j,i) =
          std::log(pres * std::pow(idens, gamma_ad)); iuov++;
        // for sound speed
        Real sound = std::sqrt(gamma_ad * pres * idens);
        user_out_var(iuov,k,j,i) = sound; iuov++;

        // for gacc1
        user_out_var(iuov,k,j,i) = ruser_meshblock_data[UGRAVD](0,k,j,i); iuov++;
        // for gacc2
        user_out_var(iuov,k,j,i) = ruser_meshblock_data[UGRAVA](0,k,j,i); iuov++;

        // for centrifugal force
        user_out_var(iuov,k,j,i) = ruser_meshblock_data[UCENTRI](0,k,j,i); iuov++;
        // for coriolis force
        user_out_var(iuov,k,j,i) = ruser_meshblock_data[UCORIOL](0,k,j,i); iuov++;

        // for Mach number
        Real v_sq = SQR(vx) + SQR(vy) + SQR(vz);
        Real mach = std::sqrt(v_sq) / sound;
        user_out_var(iuov,k,j,i) = mach; iuov++;
      }
    }
  }
}



void UserBNDInnerX1(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,is-i) = prim(IDN,k,j,is);
        prim(IVX,k,j,is-i) = prim(IVX,k,j,is);
        prim(IVY,k,j,is-i) = prim(IVY,k,j,is);
        prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is);
        prim(IPR,k,j,is-i) = prim(IPR,k,j,is);
      }
    }
  }
  return;
}

void UserBNDOuterX1(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie);
        prim(IVX,k,j,ie+i) = prim(IVX,k,j,ie);
        prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
        prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
        prim(IPR,k,j,ie+i) = prim(IPR,k,j,ie);
      }
    }
  }
  return;
}

void UserBNDInnerX2(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=ks; k<=ke; k++) {
    for (int j=1; j<=ngh; j++) {
      for (int i=is; i<=ie; i++) {
        prim(IDN,k,js-j,i) = prim(IDN,k,js,i);
        prim(IVX,k,js-j,i) = prim(IVX,k,js,i);
        prim(IVY,k,js-j,i) = prim(IVY,k,js,i);
        prim(IVZ,k,js-j,i) = prim(IVZ,k,js,i);
        prim(IPR,k,js-j,i) = prim(IPR,k,js,i);
      }
    }
  }
  return;
}

void UserBNDOuterX2(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=ks; k<=ke; k++) {
    for (int j=1; j<=ngh; j++) {
      for (int i=is; i<=ie; i++) {
        prim(IDN,k,je+j,i) = prim(IDN,k,je,i);
        prim(IVX,k,je+j,i) = prim(IVX,k,je,i);
        prim(IVY,k,je+j,i) = prim(IVY,k,je,i);
        prim(IVZ,k,je+j,i) = prim(IVZ,k,je,i);
        prim(IPR,k,je+j,i) = prim(IPR,k,je,i);
      }
    }
  }
  return;
}

void UserBNDInnerX3(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=1; k<=ngh; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        prim(IDN,ks-k,j,i) = prim(IDN,ks,j,i);
        prim(IVX,ks-k,j,i) = prim(IVX,ks,j,i);
        prim(IVY,ks-k,j,i) = prim(IVY,ks,j,i);
        prim(IVZ,ks-k,j,i) = prim(IVZ,ks,j,i);
        prim(IPR,ks-k,j,i) = prim(IPR,ks,j,i);
      }
    }
  }
  return;
}

void UserBNDOuterX3(MeshBlock *pmb, Coordinates *pco,
  AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=1; k<=ngh; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        prim(IDN,ke+k,j,i) = prim(IDN,ke,j,i);
        prim(IVX,ke+k,j,i) = prim(IVX,ke,j,i);
        prim(IVY,ke+k,j,i) = prim(IVY,ke,j,i);
        prim(IVZ,ke+k,j,i) = prim(IVZ,ke,j,i);
        prim(IPR,ke+k,j,i) = prim(IPR,ke,j,i);
      }
    }
  }
  return;
}



void SourceStar(MeshBlock *pmb, const Real time, const Real dt,
                const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                AthenaArray<Real> &cons_scalar) {
  if (add_coriolis) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          // update coriolis force
          pmb->ruser_meshblock_data[UCORIOL](0,k,j,i) =  2.0 * Omega_orbit * prim(IVY,k,j,i);
          pmb->ruser_meshblock_data[UCORIOL](1,k,j,i) = -2.0 * Omega_orbit * prim(IVX,k,j,i);
          pmb->ruser_meshblock_data[UCORIOL](2,k,j,i) =  0.0;
        }
      }
    }
  }
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &den  = prim(IDN,k,j,i);

        for (int d=0; d<3; d++) {
          Real acc_pre = pmb->ruser_meshblock_data[UGRAVD](d,k,j,i)
                       + pmb->ruser_meshblock_data[UGRAVA](d,k,j,i)
                       + pmb->ruser_meshblock_data[UCENTRI](d,k,j,i);
          cons(IM1+d,k,j,i) += dt*den
                             *(acc_pre + pmb->ruser_meshblock_data[UCORIOL](d,k,j,i));

          // for the energy source term
          cons(IEN,k,j,i) += dt*den*acc_pre*prim(IVX+d,k,j,i);
        }
      }
    }
  }
  // pmb->iuser_meshblock_data[TSTEP_COUNTER](0)++;
  // pmb->iuser_meshblock_data[TSTEP_COUNTER](0) %= rk_cycle;
}

Real CalcDensity(Real r) {
  Real rho_res = 1.0;
  return rho_res;
}

Real CalcTemperature(Real r) {
  Real temp_res = 1.0;
  return temp_res;
}

// Real CalcDonorGravity(Real x, Real y, Real z) {
//   Real r = std::sqrt(SQR(x+separation_sim)+SQR(y)+SQR(z)+grav_eps);
//   return -GMtot_sim/(1.0+mass_ratio)/SQR(r);
// }

// Real CalcDonorGravityPot(Real x, Real y, Real z) {
//   Real r = std::sqrt(SQR(x+separation_sim)+SQR(y)+SQR(z)+grav_eps);
//   return -GMtot_sim/(1.0+mass_ratio)/r;
// }

// Real CalcAccretorGravity(Real r) {
//   Real r_acc = std::sqrt(SQR(r)+grav_eps);
//   return -GMtot_sim*mass_ratio/(1.0+mass_ratio)/SQR(r_acc);
// }

// Real CalcAccretorGravityPot(Real r) {
//   Real r_acc = std::sqrt(SQR(r)+grav_eps);
//   return -GMtot_sim*mass_ratio/(1.0+mass_ratio)/r_acc;
// }

Real CalcRochePotentialPre(Real x1) {
  // origin at donor
  Real r1 = std::sqrt(SQR(x1)+TINY_NUMBER);
  Real r2 = std::sqrt(SQR(x1-separation_sim)+TINY_NUMBER);
  Real x1_CM = (mass_ratio*separation_sim) / (1.0 + mass_ratio);
  Real rcyl = x1-x1_CM;
  Real M1 = Mtot_sim / (1.0 + mass_ratio);
  Real M2 = Mtot_sim - M1;
  Real phi = -M1/r1 - M2/r2 - 0.5*SQR(rcyl*Omega_orbit);
  return phi;
}

// Real CalcRochePotential(MeshBlock *pmb, Real x1, Real x2, Real x3) {
//   // origin at accretor
//   Real r1 = std::sqrt(SQR(x1 + separation_sim) + SQR(x2) + SQR(x3) + TINY_NUMBER);
//   Real r2 = std::sqrt(SQR(x1) + SQR(x2) + SQR(x3) + TINY_NUMBER);
//   Real x1_CM = (mass_ratio*separation_sim) / (1.0 + mass_ratio);
//   Real rcyl_sq = SQR(x1-x1_CM) + SQR(x2-x2_CM);

//   Real M1 = Mtot_sim / (1.0 + mass_ratio);
//   Real M2 = Mtot_sim - M1;
//   Real phi = -M1/r1 - M2/r2 - 0.5*SQR(rcyl*Omega_orbit); // assuming G=1
//   return phi;
// }

void FindLocalMaximum(Real x_start, Real x_end,
                      Real &x_max, Real &phi_max) {
  const int N = 1000;
  Real dx = (x_end - x_start) / N;
  x_max = -1.0;
  phi_max = -1e60;
  for (int i = 0; i < N; i++) {
    Real x = x_start + ((double) i + 0.5) * dx;
    Real phi_roche = CalcRochePotentialPre(x);
    if (phi_roche > phi_max) {
      phi_max = phi_roche;
      x_max = x;
    }
  }
  return;
}


void TestForRoche(Real (&x_Ls)[3], Real (&phi_Ls)[3]) {
  // L1
  FindLocalMaximum(0.0, separation_sim,
                   x_Ls[0], phi_Ls[0]);
  // L2
  FindLocalMaximum(separation_sim, 2.0*separation_sim,
                   x_Ls[1], phi_Ls[1]);
  // L3
  FindLocalMaximum(-separation_sim, 0.0,
                   x_Ls[2], phi_Ls[2]);
  
  std::cout << "L1: x=" << x_Ls[0] << ", phi=" << phi_Ls[0] << std::endl;
  std::cout << "L2: x=" << x_Ls[1] << ", phi=" << phi_Ls[1] << std::endl;
  std::cout << "L3: x=" << x_Ls[2] << ", phi=" << phi_Ls[2] << std::endl;

  std::cout << "Transform to origin at accretor:\n";
  for (int i = 0; i < 3; i++) x_Ls[i] -= separation_sim;

  return;
}