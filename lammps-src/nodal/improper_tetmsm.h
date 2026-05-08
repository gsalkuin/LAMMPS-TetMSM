/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Gabriel Alkuino (Syracuse University)

   improper_style tetmsm

   Nodal-averaged tetrahedral volume penalty for mass-spring models on
   tetrahedral meshes.  Nodal reference volumes V0,i are computed from
   the initial geometry and stored in binary restart files.

   Nodal volume:  V_i = (1/4) sum_adj V_tet
   Strain:        eps_i = (V_i - V0,i) / V0,i
   Energy:        U = sum_i (1/2) kappa_V * V0,i * eps_i^2
   Forces:        for each tet, sum kappa_V * eps_i over its 4 nodes and
                 apply to the 4 vertices via the tetra volume gradients

   where kappa_V = G * (4*nu - 1) / (1 - 2*nu) is computed internally
   from the user-supplied shear modulus G and Poisson's ratio nu.

   Coeffs:   improper_coeff TYPE G nu
------------------------------------------------------------------------- */

#ifdef IMPROPER_CLASS
// clang-format off
ImproperStyle(tetmsm,ImproperTetMSM);
// clang-format on
#else

#ifndef LMP_IMPROPER_TETMSM_H
#define LMP_IMPROPER_TETMSM_H

#include "improper.h"

#include <cstdint>

namespace LAMMPS_NS {

class FixPropertyAtom;
class FixTetMSMComm;

class ImproperTetMSM : public Improper {
 public:
  ImproperTetMSM(class LAMMPS *);
  ~ImproperTetMSM() override;
  void init_style() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  void *extract(const char *, int &) override;

 protected:
  double *G_coeff;
  double *nu_coeff;
  double *kappa_v;

  FixPropertyAtom *fix_nodalvol0;
  FixTetMSMComm *fix_comm;

  int idx_nodalvol0;
  double *nodalvol0;

  double *nodalvol;
  double *kappa_strain;
  int nmax;

  double kappa_common;
  bool kappa_common_set;

  bool built;    // true after nodal V0 initialization or restart load

  double compute_tet_volume(double **x, int i1, int i2, int i3, int i4);

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif