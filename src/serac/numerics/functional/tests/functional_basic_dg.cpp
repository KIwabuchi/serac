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

template <typename T>
void debug_sparse_matrix(serac::Functional<T>& f, double t, const mfem::Vector& U, double epsilon = 1.0e-4) {

  mfem::Vector dU(U.Size());
  dU = 0.0;

  auto [value, dfdU]                                = f(t, serac::differentiate_wrt(U));
  std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

  std::cout << "{";
  for (int i = 0; i < U.Size(); i++) {
    dU[i] = 1;
    mfem::Vector df_jvp = dfdU(dU);  // matrix-free

    std::cout << "{";
    for (int j = 0; j < df_jvp.Size(); j++) {
      std::cout << df_jvp[j];
      if (j != df_jvp.Size() - 1) {
        std::cout << ",";
      } else {
        std::cout << " ";
      }
    }
    std::cout << "}";
    if (i != U.Size() - 1) {
      std::cout << ",\n";
    } else {
      std::cout << "\n";
    }

    dU[i] = 0;
  }
  std::cout << "}" << std::endl;

  dfdU_matrix->Print("K.mtx");

}

template <int p>
void L2_test_2D()
{
  constexpr int dim = 2;
  using test_space  = L2<p, dim>;
  using trial_space = L2<p, dim>;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch2D.mesh";
  //std::string meshfile = SERAC_REPO_DIR "/data/meshes/two_tris.mesh";

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
#if 1
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
#else
        return velocity;
#endif
      }, interior_faces);

  double t = 0.0;
  check_gradient(residual, t, U);

  debug_sparse_matrix(residual, t, U);

}

//TEST(basic, L2_test_2D_linear) { L2_test_2D<1>(); }
//TEST(basic, L2_test_2D_quadratic) { L2_test_2D<2>(); }

template <int p>
void L2_test_3D()
{
  constexpr int dim = 3;
  using test_space  = L2<p, dim>;
  using trial_space = L2<p, dim>;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh";

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
#if 1
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
#else
        return velocity;
#endif
      }, interior_faces);

  double t = 0.0;
  check_gradient(residual, t, U);

  debug_sparse_matrix(residual, t, U);

}

TEST(basic, L2_test_3D_linear) { L2_test_3D<1>(); }
TEST(basic, L2_test_3D_quadratic) { L2_test_3D<2>(); }

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
