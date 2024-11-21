// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

double t = 0.0;

int num_procs, myid;

#define L2_SCALAR_SPACE

// this is an attempt to reproduce an error message described by Mike
void test()
{
  constexpr int p = 1;
  constexpr int dim = 2;

#ifdef L2_SCALAR_SPACE
  using scalar_space = serac::L2<p, 1>;
#else
  using scalar_space = serac::H1<p, 1>;
#endif
  using vector_space = serac::H1<p, dim>;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  auto L2fec = mfem::L2_FECollection(p, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace L2_fespace(pmesh.get(), &L2fec, 1, serac::ordering);

#ifdef L2_SCALAR_SPACE
  auto scalar_fec = mfem::L2_FECollection(p, dim, mfem::BasisType::GaussLobatto);
#else
  auto scalar_fec = mfem::H1_FECollection(p, dim);
#endif
  mfem::ParFiniteElementSpace scalar_fespace(pmesh.get(), &scalar_fec, 1, serac::ordering);

  auto vector_fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace vector_fespace(pmesh.get(), &vector_fec, dim, serac::ordering);

  Domain whole_domain = EntireDomain(*pmesh);

  Functional< scalar_space(scalar_space, scalar_space, vector_space, vector_space) > residual(
    &scalar_fespace, 
    {&scalar_fespace, &scalar_fespace, &vector_fespace, &vector_fespace}
  );

  residual.AddDomainIntegral(serac::Dimension<dim>{}, serac::DependsOn<0,1,2,3>{},
    [=](double time, auto /*X*/, auto Rho, auto Rho_dot, auto U0, auto UF) {
        auto U = UF * time + U0 * (1.0 - time);
        auto dx_dX = get<DERIVATIVE>(U) + Identity<dim>();
        auto dX_dx = inv(dx_dX);
 
        auto v = get<VALUE>(UF) - get<VALUE>(U0);
        auto v_X = get<DERIVATIVE>(UF) - get<DERIVATIVE>(U0);
        auto v_dx = dot(v_X, dX_dx);
        auto div_v = serac::tr(v_dx);
 
        auto rho_dot = get<VALUE>(Rho_dot);
        auto rho = get<VALUE>(Rho);
 
        auto J = det(dx_dX);
        auto JrhoV = J * dot(v, transpose(dX_dx));
        //std::string y = rho;
        //std::string z = rho*JrhoV;
        return serac::tuple{J*(rho_dot + rho*div_v), rho * JrhoV};
    },
    whole_domain
  );

}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
