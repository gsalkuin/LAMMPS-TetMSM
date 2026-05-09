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

/* ----------------------------------------------------------------------
   Contributing author: Gabriel Alkuino (Syracuse University)
------------------------------------------------------------------------- */

#include "improper_tetmsm.h"

#include "atom.h"
#include "comm.h"
#include "fix_property_atom.h"
#include "fix_tetmsm_comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

#include "fmt/format.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ImproperTetMSM::ImproperTetMSM(LAMMPS *_lmp) : Improper(_lmp)
{
  writedata = 1;
  built = false;

  fix_nodalvol0 = nullptr;
  fix_comm = nullptr;

  idx_nodalvol0 = -1;
  nodalvol0 = nullptr;

  nodalvol = nullptr;
  kappa_strain = nullptr;
  nmax = 0;

  kappa_common = 0.0;
  kappa_common_set = false;
}

/* ---------------------------------------------------------------------- */

ImproperTetMSM::~ImproperTetMSM()
{
  if (!copymode) {
    if (fix_comm) {
      if (modify->get_fix_by_id(fix_comm->id)) modify->delete_fix(fix_comm->id);
      fix_comm = nullptr;
    }
    if (fix_nodalvol0) {
      if (modify->get_fix_by_id(fix_nodalvol0->id)) modify->delete_fix(fix_nodalvol0->id);
      fix_nodalvol0 = nullptr;
    }
  }

  if (!copymode) {
    memory->destroy(nodalvol);
    memory->destroy(kappa_strain);
    nmax = 0;
  }

  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(G_coeff);
    memory->destroy(nu_coeff);
    memory->destroy(kappa_v);
  }
}

/* ---------------------------------------------------------------------- */

double ImproperTetMSM::compute_tet_volume(double **x,
                                           int i1, int i2, int i3, int i4)
{
  double ax = x[i2][0] - x[i1][0];
  double ay = x[i2][1] - x[i1][1];
  double az = x[i2][2] - x[i1][2];

  double bx = x[i3][0] - x[i1][0];
  double by = x[i3][1] - x[i1][1];
  double bz = x[i3][2] - x[i1][2];

  double cx = x[i4][0] - x[i1][0];
  double cy = x[i4][1] - x[i1][1];
  double cz = x[i4][2] - x[i1][2];

  return (ax * (by * cz - bz * cy) +
          ay * (bz * cx - bx * cz) +
          az * (bx * cy - by * cx)) / 6.0;
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::init_style()
{
  if (!allocated)
    error->all(FLERR, "Improper tetmsm: coefficients are not set (missing improper_coeff)");

  if (!force->newton_bond)
    error->all(FLERR, "Improper tetmsm (nodal): newton_bond must be on");

  // Determine if this is a single-material volume penalty (required)

  kappa_common_set = false;
  kappa_common = 0.0;

  for (int itype = 1; itype <= atom->nimpropertypes; itype++) {
    if (!setflag[itype]) continue;
    if (kappa_v[itype] == 0.0) continue;
    if (!kappa_common_set) {
      kappa_common = kappa_v[itype];
      kappa_common_set = true;
    } else {
      const double diff = fabs(kappa_v[itype] - kappa_common);
      const double scale = fmax(1.0, fabs(kappa_common));
      if (diff > 1.0e-12 * scale)
        error->all(FLERR, "Improper tetmsm: nodal volume penalty requires identical (G,nu) "
                          "for all improper types with nonzero kappa_V");
    }
  }

  // Create/reuse internal fix to store nodal reference volumes (restartable)

  const char *fixid_vol0 = "_tetmsm_nodalvol0";
  Fix *f = modify->get_fix_by_id(fixid_vol0);
  if (!f) {
    modify->add_fix(fmt::format("{} all property/atom d_nodalvol0 writedata no", fixid_vol0));
    f = modify->get_fix_by_id(fixid_vol0);
  }

  fix_nodalvol0 = dynamic_cast<FixPropertyAtom *>(f);
  if (!fix_nodalvol0)
    error->all(FLERR, "Improper tetmsm: internal fix ID {} exists but is not fix property/atom",
               fixid_vol0);

  // Create/reuse internal communication fix

  const char *fixid_comm = "_tetmsm_comm";
  Fix *fc = modify->get_fix_by_id(fixid_comm);
  if (!fc) {
    modify->add_fix(fmt::format("{} all tetmsm/comm", fixid_comm));
    fc = modify->get_fix_by_id(fixid_comm);
  }

  fix_comm = dynamic_cast<FixTetMSMComm *>(fc);
  if (!fix_comm)
    error->all(FLERR, "Improper tetmsm: internal fix ID {} exists but is not fix tetmsm/comm",
               fixid_comm);

  // Grab pointer to custom per-atom reference volume array

  int flag, ncols;
  idx_nodalvol0 = atom->find_custom("nodalvol0", flag, ncols);
  if (idx_nodalvol0 < 0)
    error->all(FLERR, "Improper tetmsm: could not find d_nodalvol0 custom atom vector");
  if (flag != 1 || ncols != 0)
    error->all(FLERR, "Improper tetmsm: d_nodalvol0 has unexpected type/shape");

  nodalvol0 = atom->dvector[idx_nodalvol0];

  // If restart claimed the reference volumes were built, but the fix did not
  // receive restart data, fall back to reinitializing from current geometry.

  if (built && !fix_nodalvol0->restart_reset) {
    if (comm->me == 0)
      error->warning(FLERR, "Improper tetmsm: nodal reference volumes were not reset from restart; "
                             "recomputing from current geometry");
    built = false;
  }
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::compute(int eflag, int vflag)
{
  int i1, i2, i3, i4, n, type;
  double f1[3], f2[3], f3[3], f4[3];
  double vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **improperlist = neighbor->improperlist;
  int nimproperlist = neighbor->nimproperlist;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_bond = force->newton_bond;

  // Quick exit if volume penalty is disabled for all types
  if (!kappa_common_set) return;

  // Allocate/grow scratch arrays
  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->grow(nodalvol, nmax, "improper:tetmsm:nodalvol");
    memory->grow(kappa_strain, nmax, "improper:tetmsm:kappa_strain");
  }

  fix_comm->set_arrays(nodalvol, kappa_strain);

  // Phase 1: accumulate nodal volumes V_i = (1/4) sum_adj V_tet
  for (int i = 0; i < nall; i++) {
    nodalvol[i] = 0.0;
    kappa_strain[i] = 0.0;
  }

  for (n = 0; n < nimproperlist; n++) {
    i1 = improperlist[n][0];
    i2 = improperlist[n][1];
    i3 = improperlist[n][2];
    i4 = improperlist[n][3];
    type = improperlist[n][4];

    if (kappa_v[type] == 0.0) continue;

    const double vol = compute_tet_volume(x, i1, i2, i3, i4);
    if (vol <= 0.0)
      error->one(FLERR, "Improper tetmsm: non-positive tetrahedron volume; check vertex ordering");
    const double vshare = 0.25 * vol;

    if (newton_bond) {
      nodalvol[i1] += vshare;
      nodalvol[i2] += vshare;
      nodalvol[i3] += vshare;
      nodalvol[i4] += vshare;
    } else {
      if (i1 < nlocal) nodalvol[i1] += vshare;
      if (i2 < nlocal) nodalvol[i2] += vshare;
      if (i3 < nlocal) nodalvol[i3] += vshare;
      if (i4 < nlocal) nodalvol[i4] += vshare;
    }
  }

  if (newton_bond) comm->reverse_comm(fix_comm);

  // Phase 2: initialize nodal reference volumes if needed
  if (!built) {
    for (int i = 0; i < nlocal; i++) {
      if (nodalvol[i] > 0.0) {
        nodalvol0[i] = nodalvol[i];
        if (nodalvol0[i] <= 0.0)
          error->one(FLERR, "Improper tetmsm: non-positive nodal reference volume; check mesh");
      } else {
        nodalvol0[i] = 0.0;
      }
    }
    built = true;
  }

  // Phase 3: compute per-node kappa*strain and nodal energy (local atoms only)
  for (int i = 0; i < nlocal; i++) {
    const double V0 = nodalvol0[i];
    if (V0 <= 0.0) continue;
    const double strain = (nodalvol[i] - V0) / V0;
    kappa_strain[i] = kappa_common * strain;

    if (eflag_global) {
      const double enode = 0.5 * kappa_common * V0 * strain * strain;
      energy += enode;
      if (eflag_atom) eatom[i] += enode;
    } else if (eflag_atom) {
      const double enode = 0.5 * kappa_common * V0 * strain * strain;
      eatom[i] += enode;
    }
  }

  // Forward communicate nodalvol and kappa_strain to ghost atoms
  comm->forward_comm(fix_comm);

  // Phase 4: compute tet forces using nodal-averaged penalty
  for (n = 0; n < nimproperlist; n++) {
    i1 = improperlist[n][0];
    i2 = improperlist[n][1];
    i3 = improperlist[n][2];
    i4 = improperlist[n][3];
    type = improperlist[n][4];

    if (kappa_v[type] == 0.0) continue;

    const double ax = x[i2][0] - x[i1][0];
    const double ay = x[i2][1] - x[i1][1];
    const double az = x[i2][2] - x[i1][2];

    const double bx = x[i3][0] - x[i1][0];
    const double by = x[i3][1] - x[i1][1];
    const double bz = x[i3][2] - x[i1][2];

    const double cx = x[i4][0] - x[i1][0];
    const double cy = x[i4][1] - x[i1][1];
    const double cz = x[i4][2] - x[i1][2];

    const double bxc_x = by * cz - bz * cy;
    const double bxc_y = bz * cx - bx * cz;
    const double bxc_z = bx * cy - by * cx;

    const double cxa_x = cy * az - cz * ay;
    const double cxa_y = cz * ax - cx * az;
    const double cxa_z = cx * ay - cy * ax;

    const double axb_x = ay * bz - az * by;
    const double axb_y = az * bx - ax * bz;
    const double axb_z = ax * by - ay * bx;

    const double sum_kappa_strain =
      kappa_strain[i1] + kappa_strain[i2] + kappa_strain[i3] + kappa_strain[i4];

    const double prefactor = -sum_kappa_strain / 24.0;

    f2[0] = prefactor * bxc_x;
    f2[1] = prefactor * bxc_y;
    f2[2] = prefactor * bxc_z;

    f3[0] = prefactor * cxa_x;
    f3[1] = prefactor * cxa_y;
    f3[2] = prefactor * cxa_z;

    f4[0] = prefactor * axb_x;
    f4[1] = prefactor * axb_y;
    f4[2] = prefactor * axb_z;

    f1[0] = -(f2[0] + f3[0] + f4[0]);
    f1[1] = -(f2[1] + f3[1] + f4[1]);
    f1[2] = -(f2[2] + f3[2] + f4[2]);

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }
    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f2[0];
      f[i2][1] += f2[1];
      f[i2][2] += f2[2];
    }
    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }
    if (newton_bond || i4 < nlocal) {
      f[i4][0] += f4[0];
      f[i4][1] += f4[1];
      f[i4][2] += f4[2];
    }

    if (vflag_either) {
      vb1x = -ax;
      vb1y = -ay;
      vb1z = -az;
      vb2x = bx - ax;
      vb2y = by - ay;
      vb2z = bz - az;
      vb3x = cx - bx;
      vb3y = cy - by;
      vb3z = cz - bz;

      ev_tally(i1, i2, i3, i4, nlocal, newton_bond, 0.0, f1, f3, f4, vb1x, vb1y, vb1z, vb2x,
               vb2y, vb2z, vb3x, vb3y, vb3z);
    }
  }
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::allocate()
{
  allocated = 1;
  const int np1 = atom->nimpropertypes + 1;

  memory->create(G_coeff, np1, "improper:G");
  memory->create(nu_coeff, np1, "improper:nu");
  memory->create(kappa_v, np1, "improper:kappa_v");
  memory->create(setflag, np1, "improper:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   improper_coeff TYPE G nu
------------------------------------------------------------------------- */

void ImproperTetMSM::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR, "Incorrect args for improper coefficients: "
                            "expected 'improper_coeff TYPE G nu'");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nimpropertypes, ilo, ihi, error);

  double G_one = utils::numeric(FLERR, arg[1], false, lmp);
  double nu_one = utils::numeric(FLERR, arg[2], false, lmp);

  if (G_one < 0.0)
    error->all(FLERR, "Improper tetmsm: G must be non-negative");
  if (nu_one >= 0.5)
    error->all(FLERR, "Improper tetmsm: nu must be less than 0.5");

  double kv_one;
  if (nu_one == 0.25)
    kv_one = 0.0;
  else
    kv_one = G_one * (4.0 * nu_one - 1.0) / (1.0 - 2.0 * nu_one);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    G_coeff[i] = G_one;
    nu_coeff[i] = nu_one;
    kappa_v[i] = kv_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for improper coefficients");
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::write_restart(FILE *fp)
{
  fwrite(&G_coeff[1], sizeof(double), atom->nimpropertypes, fp);
  fwrite(&nu_coeff[1], sizeof(double), atom->nimpropertypes, fp);
  fwrite(&kappa_v[1], sizeof(double), atom->nimpropertypes, fp);

  const int ibuilt = built ? 1 : 0;
  fwrite(&ibuilt, sizeof(int), 1, fp);
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &G_coeff[1], sizeof(double), atom->nimpropertypes,
                  fp, nullptr, error);
    utils::sfread(FLERR, &nu_coeff[1], sizeof(double), atom->nimpropertypes,
                  fp, nullptr, error);
    utils::sfread(FLERR, &kappa_v[1], sizeof(double), atom->nimpropertypes,
                  fp, nullptr, error);
  }
  MPI_Bcast(&G_coeff[1], atom->nimpropertypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&nu_coeff[1], atom->nimpropertypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&kappa_v[1], atom->nimpropertypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nimpropertypes; i++) setflag[i] = 1;

  // Backward-compatible restart parsing:
  // - New format: trailing int built flag (0/1)
  // - Old format: trailing int nentries (typically > 1) followed by per-tet V0 map

  int marker = 0;
  if (comm->me == 0) utils::sfread(FLERR, &marker, sizeof(int), 1, fp, nullptr, error);
  MPI_Bcast(&marker, 1, MPI_INT, 0, world);

  if (marker <= 1) {
    built = (marker != 0);
  } else {
    const int nentries = marker;
    if (comm->me == 0) {
      for (int i = 0; i < nentries; i++) {
        int64_t t1, t2, t3, t4;
        double v0;
        utils::sfread(FLERR, &t1, sizeof(int64_t), 1, fp, nullptr, error);
        utils::sfread(FLERR, &t2, sizeof(int64_t), 1, fp, nullptr, error);
        utils::sfread(FLERR, &t3, sizeof(int64_t), 1, fp, nullptr, error);
        utils::sfread(FLERR, &t4, sizeof(int64_t), 1, fp, nullptr, error);
        utils::sfread(FLERR, &v0, sizeof(double), 1, fp, nullptr, error);
      }
    }

    // Legacy restarts stored a per-tet V0 map; nodal reference volumes will
    // be recomputed unless they also exist in the fix property/atom restart.
    built = true;
  }
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nimpropertypes; i++)
    fprintf(fp, "%d %g %g\n", i, G_coeff[i], nu_coeff[i]);
}

/* ---------------------------------------------------------------------- */

void *ImproperTetMSM::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "G") == 0) return (void *) G_coeff;
  if (strcmp(str, "nu") == 0) return (void *) nu_coeff;
  if (strcmp(str, "kappa_v") == 0) return (void *) kappa_v;
  return nullptr;
}