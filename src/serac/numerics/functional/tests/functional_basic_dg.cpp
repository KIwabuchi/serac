// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

template <int dim, int p>
void L2_test(std::string meshfile)
{
  using test_space  = L2<p, dim>;
  using trial_space = L2<p, dim>;

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 0);

  auto fec = mfem::L2_FECollection(p, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec, dim, serac::ordering);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Construct the new functional object using the specified test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  constexpr int DERIVATIVE = 1;

  Domain interior_faces = InteriorFaces(*mesh);

  residual.AddInteriorFaceIntegral(
      Dimension<dim-1>{}, DependsOn<0>{},
      [=](double /*t*/, auto X, auto velocity) {
        // compute the surface normal
        auto dX_dxi = get<DERIVATIVE>(X);
        auto n = normalize(cross(dX_dxi));

        // extract the velocity values from each side of the interface
        // note: the orientation convention is such that the normal 
        //       computed as above will point from from side 1->2
        auto [u_1, u_2] = velocity; 

        auto a = dot(u_2 - u_1, n);

        auto f_1 = u_1 * a;
        auto f_2 = u_2 * a;
        return serac::tuple{f_1, f_2};
      }, interior_faces);

  double t = 0.0;
  check_gradient(residual, t, U);

}

TEST(basic, L2_test_tris_and_quads_linear) { L2_test<2, 1>(SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh"); }
TEST(basic, L2_test_tris_and_quads_quadratic) { L2_test<2, 2>(SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh"); }

TEST(basic, L2_test_tets_linear) { L2_test<3, 1>(SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }
TEST(basic, L2_test_tets_quadratic) { L2_test<3, 2>(SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }

TEST(basic, L2_test_hexes_linear) { L2_test<3, 1>(SERAC_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }
TEST(basic, L2_test_hexes_quadratic) { L2_test<3, 2>(SERAC_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }

template <int dim, int p>
void L2_qoi_test(std::string meshfile)
{
  using trial_space = L2<p, dim>;

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 0);

  auto fec = mfem::L2_FECollection(p, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec, dim, serac::ordering);

  int seed = 0;
  mfem::HypreParVector U = *fespace.NewTrueDofVector();
  U.Randomize(seed);

  // Construct the new functional object using the specified test and trial spaces
  Functional<double(trial_space)> qoi({&fespace});

  constexpr int DERIVATIVE = 1;

  Domain interior_faces = InteriorFaces(*mesh);

  qoi.AddInteriorFaceIntegral(
      Dimension<dim-1>{}, DependsOn<0>{},
      [=](double /*t*/, auto X, auto velocity) {
        // compute the unit surface normal
        auto dX_dxi = get<DERIVATIVE>(X);
        auto n = normalize(cross(dX_dxi));

        // extract the velocity values from each side of the interface
        // note: the orientation convention is such that the normal 
        //       computed as above will point from from side 1->2
        auto [u_1, u_2] = velocity; 

        auto a = dot(u_2 - u_1, n);

        auto f_1 = u_1 * a;
        auto f_2 = u_2 * a;
        return dot(f_1, f_2);
      }, interior_faces);

  double t = 0.0;
  check_gradient(qoi, t, U);

}

TEST(basic, L2_qoi_test_tri_and_quads_linear) { L2_qoi_test<2, 1>(SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh"); }
TEST(basic, L2_qoi_test_tri_and_quads_quadratic) { L2_qoi_test<2, 2>(SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh"); }

TEST(basic, L2_qoi_test_tets_linear) { L2_qoi_test<3, 1>(SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }
TEST(basic, L2_qoi_test_tets_quadratic) { L2_qoi_test<3, 2>(SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }

TEST(basic, L2_qoi_test_hexes_linear) { L2_qoi_test<3, 1>(SERAC_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }
TEST(basic, L2_qoi_test_hexes_quadratic) { L2_qoi_test<3, 2>(SERAC_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }

int main(int argc, char* argv[])
{
  int num_procs, myid;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
