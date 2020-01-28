// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef CONDUCTION_SOLVER
#define CONDUCTION_SOLVER

#include "mfem.hpp"
#include "base_solver.hpp"

class DynamicConductionOperator;

class ThermalSolver : public BaseSolver
{
protected:
  mfem::ParBilinearForm *m_M_form;
  mfem::ParBilinearForm *m_K_form;
  mfem::HypreParMatrix *m_M_mat;
  mfem::HypreParMatrix *m_K_mat; // T = M + dt K

  mfem::ParLinearForm *m_l_form;
  mfem::HypreParVector *m_rhs;

  mfem::CGSolver *m_K_solver;    // Krylov solver for inverting the stiffness matrix K
  mfem::HypreSmoother *m_K_prec; // Preconditioner for the stiffness matrix K

  mfem::Coefficient *m_kappa; // Conduction coefficient
  mfem::Coefficient *m_source;

  DynamicConductionOperator *m_dyn_oper;

  LinearSolverParameters m_lin_params;

  bool m_dynamic;
  bool m_gf_initialized;

public:
  ThermalSolver(int order, mfem::ParMesh *pmesh);

  void SetTemperatureBCs(mfem::Array<int> &temp_bdr, mfem::Coefficient *temp_bdr_coef);

  void SetFluxBCs(mfem::Array<int> &flux_bdr, mfem::Coefficient *flux_bdr_coef);

  void StaticSolve();

  void AdvanceTimestep(double dt);

  void SetConductivity(mfem::Coefficient &kappa);

  void SetInitialState(mfem::Coefficient &temp);

  void SetSource(mfem::Coefficient &source);

  void CompleteSetup(const bool allow_dynamic = true);

  void SetLinearSolverParameters(const LinearSolverParameters &params);

  virtual ~ThermalSolver();
};

// After spatial discretization, the conduction model can be written as:
//
//     du/dt = M^{-1}(-Ku)
//
//  where u is the vector representing the temperature, M is the mass matrix,
//  and K is the diffusion opeperator.
//
//  Class ConductionSolver represents the right-hand side of the above ODE.

class DynamicConductionOperator : public mfem::TimeDependentOperator
{
protected:

  mfem::CGSolver *m_M_solver;    // Krylov solver for inverting the mass matrix M
  mfem::CGSolver *m_T_solver;    // Implicit solver for T = M + dt K
  mfem::HypreSmoother *m_M_prec; // Preconditioner for the mass matrix M
  mfem::HypreSmoother *m_T_prec; // Preconditioner for the implicit solver

  mfem::HypreParMatrix *m_M_mat;
  mfem::HypreParMatrix *m_K_mat;
  mfem::HypreParMatrix *m_T_mat; // T = M + dt K

  mfem::Vector *m_true_rhs;

  mutable mfem::Vector m_z; // auxiliary vector

public:
  DynamicConductionOperator(MPI_Comm comm, int height, LinearSolverParameters &params);

  void SetMMatrix(mfem::HypreParMatrix *M_mat);
  void SetKMatrixAndRHS(mfem::HypreParMatrix *K_mat, mfem::Vector *rhs);

  virtual void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const;
  /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
      This is the only requirement for high-order SDIRK implicit integration.*/
  virtual void ImplicitSolve(const double dt, const mfem::Vector &u, mfem::Vector &k);

  virtual ~DynamicConductionOperator();

};


#endif
