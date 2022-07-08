// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_mechanics_functional.hpp"
#include "serac/physics/materials/thermal_functional_material.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

template <int p>
void functional_test_static_3D(double expected_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim                 = 3;
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the thermal solver configurations
  auto thermal_options = Thermal::defaultQuasistaticOptions();

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions solid_options = {default_linear_options, default_nonlinear_options};

  // Construct a functional-based thermal-solid solver
  // BT 04/27/2022 This can't be instantiated yet.
  // The material model needs to be implemented before this
  // module can be used.
  ThermalMechanicsFunctional<p, dim> thermal_solid_solver(thermal_options, solid_options, GeometricNonlinearities::On,
                                                          FinalMeshOption::Deformed, "thermal_solid_functional");

  double                                rho       = 1.0;
  double                                E         = 1.0;
  double                                nu        = 0.25;
  double                                c         = 1.0;
  double                                alpha     = 1.0e-3;
  double                                theta_ref = 1.0;
  double                                k         = 1.0;
  GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};
  thermal_solid_solver.setMaterial(material);

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  thermal_solid_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solid_solver.setTemperature(one);

  // Define the function for the disolacement boundary condition
  auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };

  // Set the initial displcament and boundary condition
  thermal_solid_solver.setDisplacementBCs(ess_bdr, zeroVector);
  thermal_solid_solver.setDisplacement(zeroVector);

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  thermal_solid_solver.initializeOutput(serac::OutputType::VisIt, "thermal_mechanics_without_input_file_output");

  // dump initial state to output
  //thermal_solid_solver.outputState();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solid_solver.advanceTimestep(dt);

  // thermal_solid_solver.outputState();

  // thermal_solid_solver.temperature().gridFunc().Print();

  EXPECT_NEAR(expected_norm, norm(thermal_solid_solver.displacement()), 1.0e-6);

  // Check the final temperature norm
  double temperature_norm_exact = 2.0 * std::sqrt(2.0);
  EXPECT_NEAR(temperature_norm_exact, norm(thermal_solid_solver.temperature()), 1.0e-6);
}

template <int p>
void functional_test_shrinking_3D(double expected_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim                 = 3;
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // Define a boundary attribute set
  std::set<int> constraint_bdr = {1};
  std::set<int> temp_bdr = {1,2,3};

  // define the thermal solver configurations
  auto thermal_options = Thermal::defaultQuasistaticOptions();

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions solid_options = {default_linear_options, default_nonlinear_options};

  ThermalMechanicsFunctional<p, dim> thermal_solid_solver(thermal_options, solid_options,
                                                          GeometricNonlinearities::On,
                                                          FinalMeshOption::Deformed,
                                                          "thermal_solid_functional");

  double                                rho       = 1.0;
  double                                E         = 1.0;
  double                                nu        = 0.0;
  double                                c         = 1.0;
  double                                alpha     = 1.0e-3;
  double                                theta_ref = 2.0;
  double                                k         = 1.0;
  GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};
  thermal_solid_solver.setMaterial(material);

  // Define the function for the initial temperature
  double theta_0 = 1.0;
  auto initial_temperature_field = [theta_0](const mfem::Vector&, double) -> double { return theta_0; };

  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  //thermal_solid_solver.setTemperatureBCs(ess_bdr, theta_0);
  thermal_solid_solver.setTemperatureBCs(temp_bdr, one);
  thermal_solid_solver.setTemperature(initial_temperature_field);

  // Define the function for the disolacement boundary condition
  auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };

  // Set the initial displacement and boundary condition
  thermal_solid_solver.setDisplacementBCs(constraint_bdr, zeroVector);
  thermal_solid_solver.setDisplacement(zeroVector);

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  thermal_solid_solver.initializeOutput(serac::OutputType::VisIt, "thermal_mechanics_without_input_file_output");

  // dump initial state to output
  //thermal_solid_solver.outputState();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solid_solver.advanceTimestep(dt);

  //thermal_solid_solver.outputState();

  // Check the final displacement norm
  EXPECT_NEAR(expected_norm, norm(thermal_solid_solver.displacement()), 1.0e-4);
}

template <int p>
void parameterized(double expected_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  mfem::DenseMatrix A(3);
  A = 0.0;
  A(0, 0) = 0.523421770118331;
  A(0, 1) = 0.207205376077508;
  A(0, 2) = 0.600042309223256;
  A(1, 0) = 0.437180599730879;
  A(1, 1) = 0.095947283836495;
  A(1, 2) = 0.017796825926619;
  A(2, 0) = 0.149663987551694;
  A(2, 1) = 0.845137263999642;
  A(2, 2) = 0.594085227873111;
  mfem::Vector b(3);
  b(0) = 0.072393541428884;
  b(1) = 0.020326864245481;
  b(2) = 0.077181916474764;
  auto exact_displacement_function = [&A, &b](const mfem::Vector& X, mfem::Vector& u) {
    A.Mult(X, u);
    u += b;
  };

  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  // This mesh leads to perfectly straight-sided hexes, which
  // leaves the Jacobians of the element transformations constant.
  // The test should be made stronger by having non-constant
  // Jacobians. For a problem with tractions, the surface
  // facets should be non-affine as well.
  auto mesh = mesh::refineAndDistribute(buildCuboidMesh(4, 4, 4, 0.25, 0.25, 0.25),
                                        serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  // define the thermal solver configurations
  auto thermal_options = Thermal::defaultQuasistaticOptions();

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                         .abs_tol     = 1.0e-10,
                                                         .print_level = 0,
                                                         .max_iter    = 500,
                                                         .lin_solver  = LinearSolver::GMRES,
                                                         .prec        = HypreBoomerAMGPrec{}};

  const NonlinearSolverOptions default_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 10, .print_level = 1};

  const typename solid_util::SolverOptions solid_options = {default_linear_options, default_nonlinear_options};

  
  ThermalMechanicsFunctional<p, dim, H1<p> > thermal_solid_solver(thermal_options, solid_options,
                                                                  GeometricNonlinearities::On,
                                                                  FinalMeshOption::Deformed,
                                                                  "thermal_solid_functional");

  double                                rho       = 1.0;
  double                                E         = 1.0;
  double                                nu        = 0.0;
  double                                c         = 1.0;
  double                                alpha0    = 0.0;//1.0e-3;
  double                                theta_ref = 2.0;
  double                                k         = 1.0;
  ParameterizedGreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha0, theta_ref, k};
  thermal_solid_solver.setMaterial(material);

  // parameterize the coefficient of thermal expansion
  FiniteElementState thermal_expansion_scaling(
      StateManager::newState(FiniteElementState::Options{.order = p, .name = "CTE scale"}));
  thermal_expansion_scaling = 1.5;

  std::function<double(const mfem::Vector&, double)> f = [](const mfem::Vector& /*x*/, double /*t*/) {
    return 1.0;//1.0 - x[0] * 0.5;
  };
  mfem::FunctionCoefficient coef(f);
  thermal_expansion_scaling.project(coef);
  thermal_solid_solver.setParameter(thermal_expansion_scaling, 0);

  // Define a boundary attribute set
  std::set<int> constraint_bdr = {1, 2, 3, 4, 5, 6};
  std::set<int> temp_bdr = {1, 2, 3, 4, 5, 6};
  
  // Set a constant initial temperature, at the thermal expansion reference temperature value.
  auto initial_temperature_field = [theta_ref](const mfem::Vector&, double) -> double { return theta_ref; };
  thermal_solid_solver.setTemperature(initial_temperature_field);

  // set a constant temperature boundary condition
  // set it below the reference temperature
  auto one = [theta_ref](const mfem::Vector&, double) -> double { return theta_ref - 1.0; };
  thermal_solid_solver.setTemperatureBCs(temp_bdr, one);

  // displacement boundary condition
  thermal_solid_solver.setDisplacementBCs(constraint_bdr, exact_displacement_function);

  // Cheating initial guess
  auto near_exact_displacement_function = [&A, &b](const mfem::Vector& X, mfem::Vector& u) {
    A.Mult(X, u);
    u += b;
    u *= 0.99;
  };
  thermal_solid_solver.setDisplacement(near_exact_displacement_function);

  // double G = 0.5*E/(1.0 + nu);
  // double K = E/3.0/(1.0 - 2.0*nu);
  // thermal_solid_solver.addBodyForce(
  //     [K, G, alpha0](const tensor<double, 3>& /*x*/, const double /*t*/, const auto& /*u*/, const auto& /*dudx*/) {
  //       double dummy = K*G*alpha0;
  //       return tensor<double, 3>{{0.0, 0.0, 0.0}};
  //     });
  //{{6.0*K*((1.0 - x[0]*0.5)*alpha0 - 0.5*x[0]) - 8.0/3.0*G - 2.0*K, 0.0, 0.0}};
  
  thermal_solid_solver.completeSetup();

  thermal_solid_solver.initializeOutput(serac::OutputType::VisIt, "thermal_mechanics_without_input_file_output");
  // dump initial state to output
  thermal_solid_solver.outputState("pv_output");

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solid_solver.advanceTimestep(dt);

  thermal_solid_solver.outputState();

  // Compute norm of error in numerical solution
  // auto exact_solution_coef = std::make_shared<mfem::VectorFunctionCoefficient>(
  //     dim, exact_displacement_function);
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement_function);
  double error_norm = thermal_solid_solver.displacement().gridFunc().ComputeL2Error(exact_solution_coef);

  std::cout << expected_norm << std::endl;
  
  EXPECT_LT(error_norm, 1e-10);
}


}  // namespace serac

TEST(Thermomechanics, staticTest)
{
  constexpr int p = 2;
  serac::functional_test_static_3D<p>(0.0);
}

TEST(Thermomechanics, thermalContraction)
{
  constexpr int p = 2;
  // this is the small strain solution, which works with a loose enought tolerance
  // TODO work out the finite deformation solution
  double alpha = 1e-3;
  double L = 8;
  double delta_theta = 1.0;
  serac::functional_test_shrinking_3D<p>(std::sqrt(L*L*L/3.0)*alpha*delta_theta);
}

TEST(Thermomechanics, parameterized)
{
  constexpr int p = 2;
  // this is the small strain solution, which works with a loose enought tolerance
  // TODO work out the finite deformation solution
  double alpha = 1.5e-3;
  double L = 8;
  double delta_theta = 1.0;
  serac::parameterized<p>(std::sqrt(L*L*L/3.0)*alpha*delta_theta);
}

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
