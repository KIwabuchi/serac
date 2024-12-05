#include <gtest/gtest.h>

#include <fstream>

#include "mfem.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "serac/numerics/equation_solver.hpp"
#include "serac/numerics/trust_region_solver.hpp"

#include "serac/numerics/solver_config.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"

const std::string MESHTAG = "mesh";

static constexpr int dim = 2;
static constexpr int scalar_field_order = 1;

struct MeshFixture : public testing::Test
{
  void SetUp() 
  {
    serac::StateManager::initialize(datastore, "hydro_dynamics");

    auto mfem_shape = mfem::Element::QUADRILATERAL;

    double length = 0.5; double width = 2.0;
    auto meshtmp = serac::mesh::refineAndDistribute(
          mfem::Mesh::MakeCartesian2D(2, 1, mfem_shape, true, length, width),
         0, 0);
      mesh = &serac::StateManager::setMesh(std::move(meshtmp), MESHTAG);
  }

  axom::sidre::DataStore datastore;
  mfem::ParMesh* mesh;
};

std::vector<serac::FiniteElementState> applyLinearOperator(const Mat& A, const std::vector<serac::FiniteElementState>& states)
{
  std::vector<serac::FiniteElementState> Astates;
  for (auto s : states) {
    Astates.emplace_back(s);
  }

  int local_rows(states[0].Size());
  int global_rows(GlobalSize(states[0], states[0].comm()));
  
  Vec x;
  Vec y;

  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &x);
  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &y);

  PetscInt iStart, iEnd;
  VecGetOwnershipRange(x, &iStart, &iEnd);

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  size_t num_cols = states.size();
  for (size_t c=0; c < num_cols; ++c) {
    VecSetValues(x, local_rows, &col_indices[0], &states[c][0], INSERT_VALUES);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    MatMult(A, x, y);
    VecGetValues(y, local_rows, &col_indices[0], &Astates[c][0]);
  }

  VecDestroy(&x);
  VecDestroy(&y);

  return Astates;
}


//auto createDiagonalTestMatrix(serac::FiniteElementState& x)
auto createDiagonalTestMatrix(mfem::Vector& x)
{
  const int local_rows = x.Size();
  mfem::Vector one = x; one = 1.0;
  const int global_rows = serac::GlobalSize(x, PETSC_COMM_WORLD);

  Vec b;
  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &b);

  PetscInt iStart, iEnd;
  VecGetOwnershipRange(b, &iStart, &iEnd);
  VecDestroy(&b);

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  std::vector<int> row_offsets(static_cast<size_t>(local_rows)+1);
  for (int i=0; i < local_rows+1; ++i) {
    row_offsets[static_cast<size_t>(i)] = i;
  }

  Mat A;
  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, local_rows, local_rows, global_rows, global_rows, &row_offsets[0], &col_indices[0], &x[0], &A);

  return A;
}


TEST_F(MeshFixture, QR)
{
  SERAC_MARK_FUNCTION;

  auto u1 = serac::StateManager::newState(serac::H1<scalar_field_order,1>{}, "u1", MESHTAG);
  auto u2 = serac::StateManager::newState(serac::H1<scalar_field_order,1>{}, "u2", MESHTAG);
  auto u3 = serac::StateManager::newState(serac::H1<scalar_field_order,1>{}, "u3", MESHTAG);
  auto u4 = serac::StateManager::newState(serac::H1<scalar_field_order,1>{}, "u4", MESHTAG);
  auto a = serac::StateManager::newState(serac::H1<scalar_field_order,1>{}, "a", MESHTAG);
  auto b = serac::StateManager::newState(serac::H1<scalar_field_order,1>{}, "b", MESHTAG);

  u1 = 1.0;
  for (int i=0; i < u2.Size(); ++i) {
    u2[i] = i+2;
    u3[i] = i * i - 15.0;
    u4[i] = -i + 0.1 * i*i*i - 1.0;
    a[i] = 2*i + 0.01 * i * i + 1.25;
    b[i] = -i + 0.02 * i * i + 0.1;
  }
  std::vector<serac::FiniteElementState> states = {u1,u2,u3}; //,u4};

  /*
  for (int s=0; s < states.size(); ++s) {
    for (int i=0; i < u1.Size(); ++i) {
      std::cout << states[s][i] << " ";
    }
    printf("\n");
  }

  for (int i=0; i < u1.Size(); ++i) {
    std::cout << a[i] << " ";
  }
  printf("\n");

  for (int i=0; i < u1.Size(); ++i) {
    std::cout << b[i] << " ";
  }
  printf("\n");
  */

  auto A_parallel = createDiagonalTestMatrix(a);
  std::vector<serac::FiniteElementState> Astates = applyLinearOperator(A_parallel, states);

  double delta = 3.4; //0.001;
  auto [sol, leftvecs, leftvals] = serac::solveSubspaceProblem(states, Astates, b, delta, 1);

  MatDestroy(&A_parallel);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}
