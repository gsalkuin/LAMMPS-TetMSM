/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_tetmsm_comm.h"

#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixTetMSMComm::FixTetMSMComm(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR, "Illegal fix tetmsm/comm command");

  nodalvol_ = nullptr;
  kappa_strain_ = nullptr;

  comm_forward = 2;
  comm_reverse = 1;
}

/* ---------------------------------------------------------------------- */

int FixTetMSMComm::setmask()
{
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixTetMSMComm::set_arrays(double *nodalvol, double *kappa_strain)
{
  nodalvol_ = nodalvol;
  kappa_strain_ = kappa_strain;
}

/* ---------------------------------------------------------------------- */

int FixTetMSMComm::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  if (!nodalvol_ || !kappa_strain_) error->all(FLERR, "Fix tetmsm/comm arrays are not set");

  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    buf[m++] = nodalvol_[j];
    buf[m++] = kappa_strain_[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixTetMSMComm::unpack_forward_comm(int n, int first, double *buf)
{
  if (!nodalvol_ || !kappa_strain_) error->all(FLERR, "Fix tetmsm/comm arrays are not set");

  int m = 0;
  const int last = first + n;
  for (int j = first; j < last; j++) {
    nodalvol_[j] = buf[m++];
    kappa_strain_[j] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int FixTetMSMComm::pack_reverse_comm(int n, int first, double *buf)
{
  if (!nodalvol_) error->all(FLERR, "Fix tetmsm/comm arrays are not set");

  int m = 0;
  const int last = first + n;
  for (int j = first; j < last; j++) buf[m++] = nodalvol_[j];
  return m;
}

/* ---------------------------------------------------------------------- */

void FixTetMSMComm::unpack_reverse_comm(int n, int *list, double *buf)
{
  if (!nodalvol_) error->all(FLERR, "Fix tetmsm/comm arrays are not set");

  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    nodalvol_[j] += buf[m++];
  }
}