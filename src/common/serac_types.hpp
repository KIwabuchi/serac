// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include <memory>
#include <optional>
#include <set>
#include <variant>

#include "common/logger.hpp"
#include "mfem.hpp"

namespace serac {

// Option bundling enums

enum class OutputType
{
  GLVis,
  VisIt
};

enum class TimestepMethod
{
  BackwardEuler,
  SDIRK33,
  ForwardEuler,
  RK2,
  RK3SSP,
  RK4,
  GeneralizedAlpha,
  ImplicitMidpoint,
  SDIRK23,
  SDIRK34,
  QuasiStatic
};

enum class LinearSolver
{
  CG,
  GMRES,
  MINRES
};

enum class Preconditioner
{
  Jacobi,
  BoomerAMG
};

enum class CouplingScheme
{
  OperatorSplit,
  FixedPoint,
  FullyCoupled
};

// Parameter bundles

struct LinearSolverParameters {
  double         rel_tol;
  double         abs_tol;
  int            print_level;
  int            max_iter;
  LinearSolver   lin_solver;
  Preconditioner prec;
};

struct NonlinearSolverParameters {
  double rel_tol;
  double abs_tol;
  int    max_iter;
  int    print_level;
};

// Finite element information bundle
struct FiniteElementState {
  std::shared_ptr<mfem::ParFiniteElementSpace>   space;
  std::shared_ptr<mfem::FiniteElementCollection> coll;
  std::shared_ptr<mfem::ParGridFunction>         gf;
  std::shared_ptr<mfem::Vector>                  true_vec;
  std::shared_ptr<mfem::ParMesh>                 mesh;
  std::string                                    name = "";
};

// Boundary condition information
class BoundaryCondition {
 public:
  using Coef = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

  /**
   * Constructor for setting up a boundary condition using a set of attributes
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] attrs The set of boundary condition attributes in the mesh that the BC applies to
   * @param[in] num_attrs The total number of boundary attributes for the mesh
   */
  BoundaryCondition(Coef coef, const int component, const std::set<int>& attrs, const int num_attrs = 0);

  /**
   * Minimal constructor for setting the true DOFs directly
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be -1 for all components
   * @param[in] true_dofs The indices of the relevant DOFs
   */
  BoundaryCondition(Coef coef, const int component, const mfem::Array<int>& true_dofs);

  const mfem::Array<int>& getMarkers() const { return markers_; }

  mfem::Array<int>& getMarkers() { return markers_; }

  /**
   * "Manually" set the DOF indices without specifying the field to which they apply
   * @param[in] dofs The indices of the DOFs constrained by the boundary condition
   */
  void setTrueDofs(const mfem::Array<int> dofs);

  /**
   * Uses mfem::ParFiniteElementSpace::GetEssentialTrueDofs to
   * determine the DOFs for the boundary condition
   * @param[in] state The finite element state for which the DOFs should be obtained
   */
  void setTrueDofs(FiniteElementState& state);

  const mfem::Array<int>& getTrueDofs() const
  {
    SLIC_ERROR_IF(!true_dofs_, "True DOFs only available with essential BC.");
    return *true_dofs_;
  }

  // FIXME: Temporary way of maintaining single definition of essential bdr
  // until single class created to encapsulate all BCs
  void removeAttr(const int attr) { markers_[attr - 1] = 0; }

  /**
   * Projects the boundary condition over a grid function
   * @param[inout] gf The boundary condition to project over
   * @param[in] fes The finite element space that should be used to generate
   * the scalar DOF list
   */
  void project(mfem::ParGridFunction& gf, mfem::ParFiniteElementSpace& fes) const;

  /**
   * Projects the boundary condition over a grid function
   * @pre A corresponding field (FiniteElementState) has been associated
   * with the calling object via BoundaryCondition::setTrueDofs(FiniteElementState&)
   */
  void project() const;

  /**
   * Projects the boundary condition over boundary DOFs of a grid function
   * @param[inout] gf The boundary condition to project over
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @param[in] should_be_scalar Whether the boundary condition coefficient should be a scalar coef
   */
  void projectBdr(mfem::ParGridFunction& gf, const double time, bool should_be_scalar = true) const;

  /**
   * Projects the boundary condition over boundary DOFs of a grid function
   * @param[in] time The time for the coefficient, used for time-varying coefficients
   * @param[in] should_be_scalar Whether the boundary condition coefficient should be a scalar coef
   * @pre A corresponding field (FiniteElementState) has been associated
   * with the calling object via BoundaryCondition::setTrueDofs(FiniteElementState&)
   */
  void projectBdr(const double time, bool should_be_scalar = true) const;

  /**
   * Allocates an integrator of type "Integrator" on the heap,
   * constructing it with the boundary condition's vector coefficient,
   * intended to be passed to mfem::*LinearForm::Add*Integrator
   * @return An owning pointer to the new integrator
   * @pre Requires Integrator::Integrator(mfem::VectorCoefficient&)
   */
  template <typename Integrator>
  std::unique_ptr<Integrator> newVecIntegrator() const;

  /**
   * Allocates an integrator of type "Integrator" on the heap,
   * constructing it with the boundary condition's coefficient,
   * intended to be passed to mfem::*LinearForm::Add*Integrator
   * @return An owning pointer to the new integrator
   * @pre Requires Integrator::Integrator(mfem::Coefficient&)
   */
  template <typename Integrator>
  std::unique_ptr<Integrator> newIntegrator() const;

  /**
   * Eliminates the rows and columns corresponding to the BC's true DOFS
   * from a stiffness matrix
   * @param[inout] k_mat The stiffness matrix to eliminate from,
   * will be modified.  These eliminated matrix entries can be
   * used to eliminate an essential BC to an RHS vector with
   * BoundaryCondition::eliminateToRHS
   */
  void eliminateFrom(mfem::HypreParMatrix& k_mat);

  /**
   * Eliminates boundary condition from solution to RHS
   * @param[in] k_mat_post_elim A stiffness matrix post-eliminated
   * @param[in] soln The solution vector
   * @param[out] rhs The RHS vector for the system
   * @pre BoundaryCondition::eliminateFrom has been called
   */
  void eliminateToRHS(mfem::HypreParMatrix& k_mat_post_elim, const mfem::Vector& soln, mfem::Vector& rhs);

 private:
  Coef                                  coef_;
  int                                   component_;
  mfem::Array<int>                      markers_;
  std::optional<mfem::Array<int>>       true_dofs_;  // Only if essential
  std::optional<FiniteElementState*>    state_;      // Only if essential
  std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries_;
};

template <typename Integrator>
std::unique_ptr<Integrator> BoundaryCondition::newVecIntegrator() const
{
  // Can't use std::visit here because integrators may only have a constructor accepting
  // one coef type and not the other - contained types are only known at runtime
  // One solution could be to switch between implementations with std::enable_if_t and
  // std::is_constructible_v
  SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef_),
                "Boundary condition had a non-vector coefficient when constructing an integrator.");
  return std::make_unique<Integrator>(*std::get<std::shared_ptr<mfem::VectorCoefficient>>(coef_));
}

template <typename Integrator>
std::unique_ptr<Integrator> BoundaryCondition::newIntegrator() const
{
  SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef_),
                "Boundary condition had a non-vector coefficient when constructing an integrator.");
  return std::make_unique<Integrator>(*std::get<std::shared_ptr<mfem::Coefficient>>(coef_));
}

}  // namespace serac

#endif
