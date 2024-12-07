// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

// clang-format off
constexpr tensor<double, 3, 3> A3D{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                                    {0.198344644470483, 0.060514559793513, 0.084137393813728},
                                    {0.011544253485023, 0.060942846497753, 0.186383473579596}}};

constexpr tensor<double, 3> B3D{{0.765645367640828, 0.992487355850465, 0.162199373722092}};
// clang-format on

/**
 * @brief Exact displacement solution that is an affine function
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class AffineSolution {
public:
  AffineSolution()
    : A(make_tensor<dim, dim>([](int i, int j) { return A3D[i][j]; })),
      b(make_tensor<dim>([](int i) { return B3D[i]; })) {};

  tensor<double, dim> eval(tensor<double, dim> X) const { return A*X + b; };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, mfem::Vector& u) const
  {
    auto Xt = make_tensor<dim>([&X](int i){ return X[i]; });
    auto ut = this->eval(Xt);
    for (int i = 0; i < dim; ++i) u[i] = ut[i];
  }

  /**
   * @brief Apply forcing that should produce this exact displacement
   *
   * Given the physics module, apply boundary conditions and a source
   * term that are consistent with the exact solution. This is
   * independent of the domain. The solution is imposed as an essential
   * boundary condition on the parts of the boundary identified by \p
   * essential_boundaries. On the complement of
   * \p essential_boundaries, the traction corresponding to the exact
   * solution is applied.
   *
   * @tparam p Polynomial degree of the finite element approximation
   * @tparam Material Type of the material model used in the problem
   *
   * @param material Material model used in the problem
   * @param sf The SolidMechanics module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <int p, typename Material>
  void applyLoads(const Material& material, SolidMechanics<p, dim>& sf, std::set<int> essential_boundary_attrs) const
  {
    // essential BCs
    auto ebc_func = [*this](tensor<double, dim> X, double){ return this->eval(X); };
    
    auto contains = [](const std::set<int>& my_set, int i) {
      return my_set.find(i) != my_set.end();
    };

    Domain essential_boundary = Domain::ofBoundaryElements(sf.mesh(), 
      [&essential_boundary_attrs, &contains](std::vector<tensor<double, dim>>, int attr)
      {
        return contains(essential_boundary_attrs, attr);
      }
    );
    for (int i = 0; i < dim; i++) sf.setDisplacementBCs(ebc_func, essential_boundary, i);

    // natural BCs
    typename Material::State state;
    auto H = make_tensor<dim, dim>([&](int i, int j) { return A(i,j); });
    tensor<double, dim, dim> P = material(state, H);
    auto traction = [P](auto, auto n0, auto) { return dot(P, n0); };
    sf.setTraction(traction);
  }

  const tensor<double, dim, dim> A; /// Linear part of solution. Equivalently, the displacement gradient
  const tensor<double, dim> b;      /// Constant part of solution. Rigid mody displacement.
};

/**
 * @brief Specify the kinds of boundary condition to apply
 */
enum class PatchBoundaryCondition { Essential, EssentialAndNatural };

/**
 * @brief Get boundary attributes for patch meshes on which to apply essential boundary conditions
 *
 * Parameterizes patch tests boundary conditions, as either essential
 * boundary conditions or partly essential boundary conditions and
 * partly natural boundary conditions. The return values are specific
 * to the meshes "patch2d.mesh" and "patch3d.mesh". The particular
 * portions of the boundary that get essential boundary conditions
 * are arbitrarily chosen.
 *
 * @tparam dim Spatial dimension
 *
 * @param b Kind of boundary conditions to apply in the problem
 * @return std::set<int> Boundary attributes for the essential boundary condition
 */
template <int dim>
std::set<int> essentialBoundaryAttributes(PatchBoundaryCondition bc)
{
  std::set<int> essential_boundaries;
  if constexpr (dim == 2) {
    switch (bc) {
      case PatchBoundaryCondition::Essential:
        essential_boundaries = {1, 2, 3, 4};
        break;
      case PatchBoundaryCondition::EssentialAndNatural:
        essential_boundaries = {1, 4};
        break;
    }
  } else {
    switch (bc) {
      case PatchBoundaryCondition::Essential:
        essential_boundaries = {1, 2, 3, 4, 5, 6};
        break;
      case PatchBoundaryCondition::EssentialAndNatural:
        essential_boundaries = {1, 2};
        break;
    }
  }
  return essential_boundaries;
}

/**
 * @brief Solve problem and compare numerical solution to exact answer
 *
 * @tparam element_type type describing element geometry and polynomial order to use for this test
 *
 * @param bc Specifier for boundary condition type to test
 * @return double L2 norm (continuous) of error in computed solution
 */
template < typename element_type>
double solution_error(PatchBoundaryCondition bc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_static_solve");

  constexpr int p = element_type::order;
  constexpr int dim = dimension_of(element_type::geometry);

  // BT: shouldn't this assertion be in the physics module?
  // Putting it here prevents tests from having a nonsensical spatial dimension value,
  // but the physics module should be catching this error to protect users.
  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid test");

  AffineSolution<dim> exact_displacement;

  std::string meshdir = std::string(SERAC_REPO_DIR) + "/data/meshes/";
  std::string filename;
  switch (element_type::geometry) {
    case mfem::Geometry::TRIANGLE:    filename = meshdir + "patch2D_tris.mesh"; break;
    case mfem::Geometry::SQUARE:      filename = meshdir + "patch2D_quads.mesh"; break;
    case mfem::Geometry::TETRAHEDRON: filename = meshdir + "patch3D_tets.mesh"; break;
    case mfem::Geometry::CUBE:        filename = meshdir + "patch3D_hexes.mesh"; break;
    default: SLIC_ERROR_ROOT("unsupported element type for patch test"); break;
  }
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename));

  std::string mesh_tag{"mesh_tag"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a solid mechanics solver
  serac::NonlinearSolverOptions nonlin_solver_options{.nonlin_solver = NonlinearSolver::Newton, .relative_tol = 0.0, .absolute_tol = 5.0e-14, .max_iterations = 30, .print_level=1};

  auto equation_solver = std::make_unique<EquationSolver>(nonlin_solver_options, serac::solid_mechanics::default_linear_options, pmesh.GetComm());

  SolidMechanics<p, dim> solid(std::move(equation_solver), solid_mechanics::default_quasistatic_options, "solid", mesh_tag);

  solid_mechanics::NeoHookean mat{.density=1.0, .K=1.0, .G=1.0};
  solid.setMaterial(mat);

  exact_displacement.applyLoads(mat, solid, essentialBoundaryAttributes<dim>(bc));

  // Finalize the data structures
  solid.completeSetup();

  // Perform the quasi-static solve
  solid.advanceTimestep(1.0);

  // Output solution for debugging
  // solid.outputStateToDisk("paraview_output");
  // std::cout << "displacement =\n";
  // solid.displacement().Print(std::cout);
  // std::cout << "forces =\n";
  // solid_functional.reactions().Print();

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement);
  return computeL2Error(solid.displacement(), exact_solution_coef);
}

/**
 * @brief Solve pressure-driven problem with 10% uniaxial strain and compare numerical solution to exact answer
 *
 * @tparam element_type type describing element geometry and polynomial order to use for this test
 *
 * @return double L2 norm (continuous) of error in computed solution
 */
template < typename element_type>
double pressure_error()
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_static_solve");

  constexpr int p = element_type::order;
  constexpr int dim = dimension_of(element_type::geometry);

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid test");

  auto exact_uniaxial_strain = [](tensor<double, dim> X, double /* t */) {
    tensor<double, dim> u{};
    u[0] = -0.1 * X[0];
    return u;
  };

  std::string meshdir = std::string(SERAC_REPO_DIR) + "/data/meshes/";
  std::string filename;
  switch (element_type::geometry) {
    case mfem::Geometry::TRIANGLE:    filename = meshdir + "patch2D_tris.mesh"; break;
    case mfem::Geometry::SQUARE:      filename = meshdir + "patch2D_quads.mesh"; break;
    case mfem::Geometry::TETRAHEDRON: filename = meshdir + "patch3D_tets.mesh"; break;
    case mfem::Geometry::CUBE:        filename = meshdir + "patch3D_hexes.mesh"; break;
    default: SLIC_ERROR_ROOT("unsupported element type for patch test"); break;
  }
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename));

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a solid mechanics solver
  #ifdef SERAC_USE_SUNDIALS
  serac::NonlinearSolverOptions nonlin_solver_options{.nonlin_solver = NonlinearSolver::KINBacktrackingLineSearch, .relative_tol = 0.0, .absolute_tol = 1.0e-14, .max_iterations = 30};
  #else
  serac::NonlinearSolverOptions nonlin_solver_options{.nonlin_solver = NonlinearSolver::Newton, .relative_tol = 0.0, .absolute_tol = 1.0e-14, .max_iterations = 30};
  #endif

  auto equation_solver = std::make_unique<EquationSolver>(nonlin_solver_options, serac::solid_mechanics::default_linear_options, pmesh.GetComm());

  SolidMechanics<p, dim> solid(std::move(equation_solver), solid_mechanics::default_quasistatic_options, "solid", mesh_tag);

  solid_mechanics::NeoHookean mat{.density=1.0, .K=1.0, .G=1.0};
  solid.setMaterial(mat);

  typename solid_mechanics::NeoHookean::State state;
  auto H = make_tensor<dim, dim>([](int i, int j) {
      if ( i == 0 && j == 0) {
        return -0.1;
      }
      return 0.0;
  });

  tensor<double, dim, dim> sigma = mat(state, H);
  auto P = solid_mechanics::CauchyToPiola(sigma, H);
  double pressure = -1.0 * P(0,0);

  // Set the pressure corresponding to 10% uniaxial strain
  solid.setPressure([pressure](auto&, double) {
    return pressure;
  });

  // Define the essential boundary conditions corresponding to 10% uniaxial strain everywhere
  // except the pressure loaded surface
  if constexpr (dim == 2) {
    auto set_bcs = [&solid, exact_uniaxial_strain](Domain driven, Domain fixed) {
      for (int i = 0; i < dim; ++i) solid.setDisplacementBCs(exact_uniaxial_strain, driven, i);
      solid.setFixedBCs(fixed, 1);
    };

    if (element_type::geometry == mfem::Geometry::TRIANGLE) {
      Domain driven = Domain::ofBoundaryElements(pmesh, by_attr<dim>(4));
      Domain fixed = Domain::ofBoundaryElements(pmesh, by_attr<dim>({1, 3}));
      set_bcs(driven, fixed);
    } else if (element_type::geometry == mfem::Geometry::SQUARE) {
      Domain driven = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
      Domain fixed = Domain::ofBoundaryElements(pmesh, by_attr<dim>({2, 4}));
      set_bcs(driven, fixed);
    }
  } else { // dim == 3
    auto set_bcs = [&solid, exact_uniaxial_strain](Domain driven, Domain fixed_y, Domain fixed_z) {
      for (int i = 0; i < dim; ++i) solid.setDisplacementBCs(exact_uniaxial_strain, driven, i);
      solid.setFixedBCs(fixed_y, 1);
      solid.setFixedBCs(fixed_z, 2);
    };

    Domain driven = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
    Domain fixed_y = Domain::ofBoundaryElements(pmesh, by_attr<dim>({2, 5}));
    Domain fixed_z = Domain::ofBoundaryElements(pmesh, by_attr<dim>({3, 6}));
    set_bcs(driven, fixed_y, fixed_z);
  }

  // Finalize the data structures
  solid.completeSetup();

  // Perform the quasi-static solve
  solid.advanceTimestep(1.0);

  solid.outputStateToDisk();

  // Output solution for debugging
  // solid.outputStateToDisk("paraview_output");
  // std::cout << "displacement =\n";
  // solid.displacement().Print(std::cout);
  // std::cout << "forces =\n";
  // solid_functional.reactions().Print();

  // Compute norm of error
  auto mfem_coefficient_function = [exact_uniaxial_strain](const mfem::Vector& X, mfem::Vector& u) {
    auto Xt = make_tensor<dim>([&X](int i) { return X[i]; });
    auto ut = exact_uniaxial_strain(Xt, 0.0);
    for (int i = 0; i < dim; i++) u[i] = ut[i];
  };
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, mfem_coefficient_function);
  return computeL2Error(solid.displacement(), exact_solution_coef);
}


const double tol = 1e-13;

constexpr int LINEAR = 1;
constexpr int QUADRATIC = 2;
constexpr int CUBIC = 3;

TEST(SolidMechanics, PatchTest2dQ1EssentialBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< LINEAR > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< LINEAR > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::Essential);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ1EssentialBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< LINEAR > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< LINEAR > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ2EssentialBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< QUADRATIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< QUADRATIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::Essential);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ2EssentialBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< QUADRATIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< QUADRATIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ3EssentialBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< CUBIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< CUBIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::Essential);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ3EssentialBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< CUBIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< CUBIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(hex_error, tol);
}

///////////////////////////////////////////////////////////////////////////////

TEST(SolidMechanics, PatchTest2dQ1EssentialAndNaturalBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< LINEAR > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< LINEAR > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ1EssentialAndNaturalBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< LINEAR > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< LINEAR > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ2EssentialAndNaturalBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< QUADRATIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< QUADRATIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ2EssentialAndNaturalBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< QUADRATIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< QUADRATIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ3EssentialAndNaturalBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< CUBIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< CUBIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ3EssentialAndNaturalBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< CUBIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< CUBIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ1Pressure){
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< LINEAR > >;
  double tri_error = pressure_error< triangle >();
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< LINEAR > >;
  double quad_error = pressure_error< quadrilateral >();
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ1Pressure){
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< LINEAR > >;
  double tet_error = pressure_error< tetrahedron >();
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< LINEAR > >;
  double hex_error = pressure_error< hexahedron >();
  EXPECT_LT(hex_error, tol);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
