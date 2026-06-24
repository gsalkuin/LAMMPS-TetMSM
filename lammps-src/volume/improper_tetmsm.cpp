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
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ImproperTetMSM::ImproperTetMSM(LAMMPS *_lmp) : Improper(_lmp)
{
  writedata = 1;
  built = false;
  scale = 1.0;
  scale_set = false;
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::settings(int narg, char **arg)
{
  if (scale_set)
    error->all(FLERR, "Improper tetmsm: scale is immutable once set");

  // scale must be set before improper_coeff bakes it into kappa_v
  if (allocated) {
    for (int i = 1; i <= atom->nimpropertypes; i++)
      if (setflag[i])
        error->all(FLERR, "Improper tetmsm: scale must be set before improper_coeff");
  }

  if (narg == 0) {
    scale_set = true;
    return;
  }

  if (narg != 2) error->all(FLERR, "Illegal improper_style command");
  if (strcmp(arg[0], "scale") != 0) error->all(FLERR, "Illegal improper_style command");

  scale = utils::numeric(FLERR, arg[1], false, lmp);
  if (scale < 0.0)
    error->all(FLERR, "Improper tetmsm: scale must be non-negative");

  scale_set = true;
}

/* ---------------------------------------------------------------------- */

ImproperTetMSM::~ImproperTetMSM()
{
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

double ImproperTetMSM::get_v0(int64_t t1, int64_t t2, int64_t t3, int64_t t4,
                               double **x, int i1, int i2, int i3, int i4)
{
  ImpKey key{t1, t2, t3, t4};
  auto it = v0_map.find(key);
  if (it != v0_map.end()) return it->second;

  error->one(FLERR, "Improper tetmsm: missing reference volume for tet; "
             "increase communication cutoff or check mesh");
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::compute(int eflag, int vflag)
{
  int i1, i2, i3, i4, n, type;
  double eimproper, f1[3], f2[3], f3[3], f4[3];
  double vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z;

  eimproper = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int **improperlist = neighbor->improperlist;
  int nimproperlist = neighbor->nimproperlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  if (!built) {
    std::unordered_map<ImpKey, double, ImpKeyHash> v0_local;
    v0_local.reserve(nimproperlist);

    for (int m = 0; m < nimproperlist; m++) {
      int j1 = improperlist[m][0];
      int j2 = improperlist[m][1];
      int j3 = improperlist[m][2];
      int j4 = improperlist[m][3];
      int jtype = improperlist[m][4];

      if (kappa_v[jtype] == 0.0) continue;

      double vol0 = compute_tet_volume(x, j1, j2, j3, j4);
      if (vol0 <= 0.0)
        error->one(FLERR, "Improper tetmsm: non-positive reference volume; check vertex ordering");

      v0_local[{tag[j1], tag[j2], tag[j3], tag[j4]}] = vol0;
    }

    const int nprocs = comm->nprocs;
    const int entry_bytes = 4 * sizeof(int64_t) + sizeof(double);

    int local_n = v0_local.size();
    std::vector<char> local_buf(local_n * entry_bytes);
    char *ptr = local_buf.data();
    for (auto &kv : v0_local) {
      const ImpKey &key = kv.first;
      const double val = kv.second;
      memcpy(ptr, &key.t1, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(ptr, &key.t2, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(ptr, &key.t3, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(ptr, &key.t4, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(ptr, &val, sizeof(double));     ptr += sizeof(double);
    }

    std::vector<int> counts(nprocs), displs(nprocs), byte_counts(nprocs);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

    displs[0] = 0;
    for (int i = 1; i < nprocs; i++)
      displs[i] = displs[i - 1] + counts[i - 1] * entry_bytes;
    for (int i = 0; i < nprocs; i++)
      byte_counts[i] = counts[i] * entry_bytes;

    const int total_bytes = displs[nprocs - 1] + byte_counts[nprocs - 1];
    std::vector<char> global_buf(total_bytes);
    MPI_Allgatherv(local_buf.data(), local_n * entry_bytes, MPI_CHAR,
                   global_buf.data(), byte_counts.data(), displs.data(),
                   MPI_CHAR, world);

    v0_map.clear();
    v0_map.reserve(global_buf.size() / entry_bytes);

    ptr = global_buf.data();
    const int total_entries = total_bytes / entry_bytes;
    for (int i = 0; i < total_entries; i++) {
      int64_t t1, t2, t3, t4;
      double val;
      memcpy(&t1, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(&t2, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(&t3, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(&t4, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
      memcpy(&val, ptr, sizeof(double));  ptr += sizeof(double);
      v0_map[{t1, t2, t3, t4}] = val;
    }

    built = true;
  }

  for (n = 0; n < nimproperlist; n++) {
    i1 = improperlist[n][0];
    i2 = improperlist[n][1];
    i3 = improperlist[n][2];
    i4 = improperlist[n][3];
    type = improperlist[n][4];

    if (kappa_v[type] == 0.0) continue;

    double v0 = get_v0(tag[i1], tag[i2], tag[i3], tag[i4],
                        x, i1, i2, i3, i4);

    double ax = x[i2][0] - x[i1][0];
    double ay = x[i2][1] - x[i1][1];
    double az = x[i2][2] - x[i1][2];

    double bx = x[i3][0] - x[i1][0];
    double by = x[i3][1] - x[i1][1];
    double bz = x[i3][2] - x[i1][2];

    double cx = x[i4][0] - x[i1][0];
    double cy = x[i4][1] - x[i1][1];
    double cz = x[i4][2] - x[i1][2];

    double bxc_x = by * cz - bz * cy;
    double bxc_y = bz * cx - bx * cz;
    double bxc_z = bx * cy - by * cx;

    double cxa_x = cy * az - cz * ay;
    double cxa_y = cz * ax - cx * az;
    double cxa_z = cx * ay - cy * ax;

    double axb_x = ay * bz - az * by;
    double axb_y = az * bx - ax * bz;
    double axb_z = ax * by - ay * bx;

    double vol = (ax * bxc_x + ay * bxc_y + az * bxc_z) / 6.0;

    double strain_v = (vol - v0) / v0;

    if (eflag) eimproper = 0.5 * kappa_v[type] * v0 * strain_v * strain_v;

    double prefactor = -kappa_v[type] * strain_v / 6.0;

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
      f[i1][0] += f1[0]; f[i1][1] += f1[1]; f[i1][2] += f1[2];
    }
    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f2[0]; f[i2][1] += f2[1]; f[i2][2] += f2[2];
    }
    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0]; f[i3][1] += f3[1]; f[i3][2] += f3[2];
    }
    if (newton_bond || i4 < nlocal) {
      f[i4][0] += f4[0]; f[i4][1] += f4[1]; f[i4][2] += f4[2];
    }

    if (evflag) {
      vb1x = -ax;  vb1y = -ay;  vb1z = -az;
      vb2x = bx - ax;  vb2y = by - ay;  vb2z = bz - az;
      vb3x = cx - bx;  vb3y = cy - by;  vb3z = cz - bz;

      ev_tally(i1, i2, i3, i4, nlocal, newton_bond, eimproper, f1, f3, f4,
               vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z);
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

  kv_one *= scale;

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

  int n = v0_map.size();
  fwrite(&n, sizeof(int), 1, fp);
  for (auto &[key, val] : v0_map) {
    fwrite(&key.t1, sizeof(int64_t), 1, fp);
    fwrite(&key.t2, sizeof(int64_t), 1, fp);
    fwrite(&key.t3, sizeof(int64_t), 1, fp);
    fwrite(&key.t4, sizeof(int64_t), 1, fp);
    fwrite(&val, sizeof(double), 1, fp);
  }
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

  int n = 0;
  if (comm->me == 0)
    utils::sfread(FLERR, &n, sizeof(int), 1, fp, nullptr, error);
  MPI_Bcast(&n, 1, MPI_INT, 0, world);

  const int entry_bytes = 4 * sizeof(int64_t) + sizeof(double);
  std::vector<char> buf(n * entry_bytes);
  if (comm->me == 0 && n > 0)
    utils::sfread(FLERR, buf.data(), entry_bytes, n, fp, nullptr, error);
  if (n > 0)
    MPI_Bcast(buf.data(), n * entry_bytes, MPI_CHAR, 0, world);

  v0_map.clear();
  v0_map.reserve(n);
  char *ptr = buf.data();
  for (int i = 0; i < n; i++) {
    int64_t t1, t2, t3, t4;
    double v0;
    memcpy(&t1, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&t2, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&t3, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&t4, ptr, sizeof(int64_t)); ptr += sizeof(int64_t);
    memcpy(&v0, ptr, sizeof(double));  ptr += sizeof(double);
    v0_map[{t1, t2, t3, t4}] = v0;
  }

  built = true;
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::write_restart_settings(FILE *fp)
{
  fwrite(&scale, sizeof(double), 1, fp);
}

/* ---------------------------------------------------------------------- */

void ImproperTetMSM::read_restart_settings(FILE *fp)
{
  if (comm->me == 0)
    utils::sfread(FLERR, &scale, sizeof(double), 1, fp, nullptr, error);
  MPI_Bcast(&scale, 1, MPI_DOUBLE, 0, world);
  scale_set = true;
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
  if (strcmp(str, "scale") == 0) {
    dim = 0;
    return (void *) &scale;
  }

  dim = 1;
  if (strcmp(str, "G") == 0) return (void *) G_coeff;
  if (strcmp(str, "nu") == 0) return (void *) nu_coeff;
  if (strcmp(str, "kappa_v") == 0) return (void *) kappa_v;
  return nullptr;
}